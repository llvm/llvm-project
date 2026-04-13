//===-- lib/runtime/trampoline.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// W^X-compliant trampoline pool implementation.
//
// This file implements a runtime trampoline pool that maintains separate
// memory regions for executable code (RX) and writable data (RW).
//
// On Linux the code region transitions RW → RX (never simultaneously W+X).
// On macOS Apple Silicon the code region uses MAP_JIT with per-thread W^X
// toggling via pthread_jit_write_protect_np, so the mapping permissions
// include both W and X but hardware enforces that only one is active at
// a time on any given thread.
//
// Architecture:
//   - Code region (RX): Contains pre-assembled trampoline stubs that load
//     callee address and static chain from a paired TDATA entry, then jump
//     to the callee with the static chain in the appropriate register.
//   - Data region (RW): Contains TrampolineData entries with {callee_address,
//     static_chain_address} pairs, one per trampoline slot.
//   - Free list: Tracks available trampoline slots for O(1) alloc/free.
//
// Thread safety: Uses Fortran::runtime::Lock (pthreads on POSIX,
// CRITICAL_SECTION on Windows) — not std::mutex — to avoid C++ runtime
// library dependence. A single global lock serializes pool operations.
// This is a deliberate V1 design choice to keep the initial W^X
// architectural change minimal. Per-thread lock-free pools are deferred
// to a future optimization patch.
//
// AddressSanitizer note: The trampoline code region is allocated via
// mmap (not malloc/new), so ASan does not track it. The data region
// and handles are allocated via malloc (through AllocateMemoryOrCrash),
// which ASan intercepts normally. No special annotations are needed.
//
// See flang/docs/InternalProcedureTrampolines.md for design details.
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/trampoline.h"
#include "flang-rt/runtime/lock.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/trampoline.h"
#include "flang/Runtime/freestanding-tools.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Platform-specific headers for memory mapping.
#if defined(_WIN32)
#include <windows.h>
#else
// On macOS/Darwin, the flang-rt CMake configuration sets
// -D_POSIX_C_SOURCE=200809, which hides BSD/Apple-specific mmap flags
// (MAP_ANON, MAP_JIT) from <sys/mman.h>. Define _DARWIN_C_SOURCE to
// re-expose them for MAP_JIT on Apple Silicon and MAP_ANON elsewhere.
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
// Some platforms (e.g. AIX) define MAP_ANON instead of MAP_ANONYMOUS.
#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#define MAP_ANONYMOUS MAP_ANON
#endif
#endif

// macOS Apple Silicon requires MAP_JIT and pthread_jit_write_protect_np
// to create executable memory under the hardened runtime.
#if defined(__APPLE__) && defined(__aarch64__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif

// Architecture support check. Stub generators exist only for x86-64 and
// AArch64. On other architectures the file compiles but the runtime API
// functions crash with a diagnostic if actually called, so that building
// flang-rt on e.g. RISC-V or PPC64 never fails.
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(_M_ARM64)
#define TRAMPOLINE_ARCH_SUPPORTED 1
#else
#define TRAMPOLINE_ARCH_SUPPORTED 0
#endif

namespace Fortran::runtime::trampoline {

/// A handle returned to the caller. Contains enough info to find
/// both the trampoline stub and its data entry.
struct TrampolineHandle {
  void *codePtr{nullptr}; // Pointer to the trampoline stub in the RX region.
  std::size_t slotIndex{0}; // Index in the pool for free-list management.
};

// Namespace-scope globals following Flang runtime conventions:
// - Lock is trivially constructible (pthread_mutex_t / CRITICAL_SECTION)
// - Pool pointer uses std::atomic for safe double-checked locking
class TrampolinePool; // Forward declaration for pointer below.
static Lock poolLock;
static std::atomic<TrampolinePool *> poolInstance{nullptr};

/// The global trampoline pool.
class TrampolinePool {
public:
  TrampolinePool() = default;

  static TrampolinePool &instance() {
    TrampolinePool *p{poolInstance.load(std::memory_order_acquire)};
    if (p) {
      return *p;
    }
    CriticalSection critical{poolLock};
    p = poolInstance.load(std::memory_order_relaxed);
    if (p) {
      return *p;
    }
    // Allocate pool using SizedNew (malloc + placement new).
    Terminator terminator{__FILE__, __LINE__};
    auto owning{SizedNew<TrampolinePool>{terminator}(sizeof(TrampolinePool))};
    p = owning.release();
    poolInstance.store(p, std::memory_order_release);
    return *p;
  }

  /// Allocate a trampoline slot and initialize it.
  TrampolineHandle *allocate(
      const void *calleeAddress, const void *staticChainAddress) {
    CriticalSection critical{lock_};
    ensureInitialized();

    if (freeHead_ == kInvalidIndex) {
      // Pool exhausted — fixed size by design for V1.
      // The pool capacity is controlled by FLANG_TRAMPOLINE_POOL_SIZE
      // (default 1024). Dynamic slab growth can be added in a follow-up
      // patch if real workloads demonstrate a need for it.
      Terminator terminator{__FILE__, __LINE__};
      terminator.Crash("Trampoline pool exhausted (max %zu slots). "
                       "Set FLANG_TRAMPOLINE_POOL_SIZE to increase.",
          poolSize_);
    }

    std::size_t index{freeHead_};
    freeHead_ = freeList_[index];

    // Initialize the data entry.
    dataRegion_[index].calleeAddress = calleeAddress;
    dataRegion_[index].staticChainAddress = staticChainAddress;

    // Create handle using SizedNew (malloc + placement new).
    Terminator terminator{__FILE__, __LINE__};
    auto owning{New<TrampolineHandle>{terminator}()};
    TrampolineHandle *handle{owning.release()};
    handle->codePtr =
        static_cast<char *>(codeRegion_) + index * kTrampolineStubSize;
    handle->slotIndex = index;

    return handle;
  }

  /// Get the callable address of a trampoline.
  void *getCallableAddress(TrampolineHandle *handle) { return handle->codePtr; }

  /// Free a trampoline slot.
  void free(TrampolineHandle *handle) {
    CriticalSection critical{lock_};

    std::size_t index{handle->slotIndex};

    // Poison the data entry so that any dangling call through a freed
    // trampoline traps immediately. Setting to NULL means the stub will
    // jump to address 0, which is unmapped on all supported platforms
    // and produces SIGSEGV/SIGBUS immediately.
    dataRegion_[index].calleeAddress = nullptr;
    dataRegion_[index].staticChainAddress = nullptr;

    // Return slot to free list.
    freeList_[index] = freeHead_;
    freeHead_ = index;

    FreeMemory(handle);
  }

private:
  static constexpr std::size_t kInvalidIndex{~std::size_t{0}};

  void ensureInitialized() {
    if (initialized_) {
      return;
    }
    initialized_ = true;

    // Check environment variable for pool size override.
    // Fixed-size pool by design (V1): avoids complexity of dynamic growth
    // and re-protection of code pages. The default (1024 slots) is
    // sufficient for typical Fortran programs. Users can override via:
    //   export FLANG_TRAMPOLINE_POOL_SIZE=4096
    if (const char *envSize = std::getenv("FLANG_TRAMPOLINE_POOL_SIZE")) {
      long val{std::strtol(envSize, nullptr, 10)};
      if (val > 0) {
        poolSize_ = {static_cast<std::size_t>(val)};
      }
    }

    // Allocate the data region (RW).
    Terminator terminator{__FILE__, __LINE__};
    dataRegion_ = static_cast<TrampolineData *>(
        AllocateMemoryOrCrash(terminator, poolSize_ * sizeof(TrampolineData)));
    runtime::memset(dataRegion_, 0, poolSize_ * sizeof(TrampolineData));

    // Allocate the code region (initially RW for writing stubs, then RX).
    std::size_t codeSize{poolSize_ * kTrampolineStubSize};
#if defined(_WIN32)
    codeRegion_ = VirtualAlloc(
        nullptr, codeSize, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#elif defined(__APPLE__) && defined(__aarch64__)
    // macOS Apple Silicon: MAP_JIT is required for pages that will become
    // executable. Use pthread_jit_write_protect_np to toggle W↔X.
    codeRegion_ = mmap(nullptr, codeSize, PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
    if (codeRegion_ == MAP_FAILED) {
      codeRegion_ = nullptr;
    }
    if (codeRegion_) {
      // Enable writing on this thread (MAP_JIT defaults to execute).
      // Guard for deployment targets older than macOS 11.0 (Apple Silicon
      // always runs >= 11.0, so this is effectively unconditional at runtime).
      if (__builtin_available(macOS 11.0, *)) {
        pthread_jit_write_protect_np(0); // 0 = writable
      }
    }
#elif defined(MAP_ANONYMOUS)
    // Linux and other POSIX platforms with MAP_ANONYMOUS.
    codeRegion_ = mmap(nullptr, codeSize, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (codeRegion_ == MAP_FAILED) {
      codeRegion_ = nullptr;
    }
#else
    // Platforms without MAP_ANONYMOUS or MAP_ANON (e.g. AIX): map /dev/zero
    // as a portable anonymous-mapping equivalent (per POSIX).
    {
      int devZero{open("/dev/zero", O_RDONLY)};
      if (devZero >= 0) {
        codeRegion_ = mmap(
            nullptr, codeSize, PROT_READ | PROT_WRITE, MAP_PRIVATE, devZero, 0);
        if (codeRegion_ == MAP_FAILED) {
          codeRegion_ = nullptr;
        }
        close(devZero);
      }
    }
#endif
    if (!codeRegion_) {
      terminator.Crash("Failed to allocate trampoline code region");
    }

    // Generate trampoline stubs.
    generateStubs();

    // Flush instruction cache. Required on architectures with non-coherent
    // I-cache/D-cache (AArch64, PPC, etc.). On x86-64 this is a no-op
    // but harmless. Without this, AArch64 may execute stale instructions.
#if defined(__APPLE__) && defined(__aarch64__)
    // On macOS, use sys_icache_invalidate (from libkern/OSCacheControl.h).
    sys_icache_invalidate(codeRegion_, codeSize);
#elif defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), codeRegion_, codeSize);
#else
    __builtin___clear_cache(static_cast<char *>(codeRegion_),
        static_cast<char *>(codeRegion_) + codeSize);
#endif

    // Make code region executable and non-writable (W^X).
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(codeRegion_, codeSize, PAGE_EXECUTE_READ, &oldProtect);
#elif defined(__APPLE__) && defined(__aarch64__)
    // Switch back to execute-only (MAP_JIT manages per-thread W^X).
    if (__builtin_available(macOS 11.0, *)) {
      pthread_jit_write_protect_np(1); // 1 = executable
    }
#else
    mprotect(codeRegion_, codeSize, PROT_READ | PROT_EXEC);
#endif

    // Initialize free list.
    freeList_ = static_cast<std::size_t *>(
        AllocateMemoryOrCrash(terminator, poolSize_ * sizeof(std::size_t)));

    for (std::size_t i{0}; i < poolSize_ - 1; ++i) {
      freeList_[i] = i + 1;
    }
    freeList_[poolSize_ - 1] = kInvalidIndex;
    freeHead_ = 0;
  }

  /// Generate platform-specific trampoline stubs in the code region.
  /// Each stub loads callee address and static chain from its paired
  /// TDATA entry and jumps to the callee.
  void generateStubs() {
#if defined(__x86_64__) || defined(_M_X64)
    generateStubsX86_64();
#elif defined(__aarch64__) || defined(_M_ARM64)
    generateStubsAArch64();
#else
    // Unsupported architecture — should never be reached because the
    // extern "C" API functions guard with TRAMPOLINE_ARCH_SUPPORTED.
    // Fill with trap bytes as a safety net.
    runtime::memset(codeRegion_, 0, poolSize_ * kTrampolineStubSize);
#endif
  }

#if defined(__x86_64__) || defined(_M_X64)
  /// Generate x86-64 trampoline stubs.
  ///
  /// Each stub does:
  ///   movabsq $dataEntry, %r11         ; load TDATA entry address
  ///   movq    8(%r11), %r10            ; load static chain -> nest register
  ///   jmpq    *(%r11)                  ; jump to callee address
  ///
  /// Total: 10 + 4 + 3 = 17 bytes, padded to kTrampolineStubSize.
  void generateStubsX86_64() {
    auto *code{static_cast<uint8_t *>(codeRegion_)};

    for (std::size_t i{0}; i < poolSize_; ++i) {
      uint8_t *stub{code + i * kTrampolineStubSize};

      // Address of the corresponding TDATA entry.
      auto dataAddr{reinterpret_cast<uint64_t>(&dataRegion_[i])};

      std::size_t off{0};

      // movabsq $dataAddr, %r11    (REX.W + B, opcode 0xBB for r11)
      stub[off++] = 0x49; // REX.WB
      stub[off++] = 0xBB; // MOV r11, imm64
      runtime::memcpy(&stub[off], &dataAddr, 8);
      off += 8;

      // movq 8(%r11), %r10         (load staticChainAddress into r10)
      stub[off++] = 0x4D; // REX.WRB
      stub[off++] = 0x8B; // MOV r/m64 -> r64
      stub[off++] = 0x53; // ModRM: [r11 + disp8], r10
      stub[off++] = 0x08; // disp8 = 8

      // jmpq *(%r11)               (jump to calleeAddress)
      stub[off++] = 0x41; // REX.B
      stub[off++] = 0xFF; // JMP r/m64
      stub[off++] = 0x23; // ModRM: [r11], opcode extension 4

      // Pad the rest with INT3 (0xCC) for safety.
      while (off < kTrampolineStubSize) {
        stub[off++] = 0xCC;
      }
    }
  }
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  /// Generate AArch64 trampoline stubs.
  ///
  /// Each stub does:
  ///   ldr x17, .Ldata_addr         ; load TDATA entry address
  ///   ldr x15, [x17, #8]           ; load static chain -> x15 (nest reg)
  ///   ldr x17, [x17]               ; load callee address
  ///   br  x17                      ; jump to callee
  ///   .Ldata_addr:
  ///     .quad <address of dataRegion_[i]>
  ///
  /// Total: 4*4 + 8 = 24 bytes, padded to kTrampolineStubSize.
  void generateStubsAArch64() {
    auto *code{static_cast<uint8_t *>(codeRegion_)};

    for (std::size_t i{0}; i < poolSize_; ++i) {
      auto *stub{reinterpret_cast<uint32_t *>(code + i * kTrampolineStubSize)};

      // Address of the corresponding TDATA entry.
      auto dataAddr{reinterpret_cast<uint64_t>(&dataRegion_[i])};

      // ldr x17, .Ldata_addr (PC-relative load, offset = 4 instructions = 16
      // bytes) LDR (literal): opc=01, V=0, imm19=(16/4)=4, Rt=17
      stub[0] = 0x58000091; // ldr x17, #16  (imm19=4, shifted left 2 = 16)
                            // Encoding: 0101 1000 0000 0000 0000 0000 1001 0001

      // ldr x15, [x17, #8]  (load static chain into x15, the nest register)
      // LDR (unsigned offset): size=11, V=0, opc=01, imm12=1(×8), Rn=17, Rt=15
      stub[1] = 0xF940062F; // ldr x15, [x17, #8]

      // ldr x17, [x17]      (load callee address)
      // LDR (unsigned offset): size=11, V=0, opc=01, imm12=0, Rn=17, Rt=17
      stub[2] = 0xF9400231; // ldr x17, [x17, #0]

      // br x17
      stub[3] = 0xD61F0220; // br x17

      // .Ldata_addr: .quad dataRegion_[i]
      runtime::memcpy(&stub[4], &dataAddr, 8);

      // Pad remaining with BRK #0 (trap) for safety.
      std::size_t usedWords{4 + 2}; // 4 instructions + 1 quad (2 words)
      for (std::size_t w{usedWords}; w < kTrampolineStubSize / sizeof(uint32_t);
          ++w) {
        stub[w] = 0xD4200000; // brk #0
      }
    }
  }
#endif

  Lock lock_;
  bool initialized_{false};
  std::size_t poolSize_{kDefaultPoolSize};

  void *codeRegion_{nullptr}; // RX after initialization
  TrampolineData *dataRegion_{nullptr}; // RW always
  std::size_t *freeList_{nullptr}; // Intrusive free list
  std::size_t freeHead_{kInvalidIndex};
};

} // namespace Fortran::runtime::trampoline

namespace Fortran::runtime {
extern "C" {

// Helper: crash with a clear message on unsupported architectures.
// This is only reached if -fsafe-trampoline was used on a target
// that lacks stub generators. The driver should emit a warning and
// ignore the flag on unsupported architectures, but the runtime
// provides a safety net.
static inline void crashIfUnsupported() {
#if !TRAMPOLINE_ARCH_SUPPORTED
  Terminator terminator{__FILE__, __LINE__};
  terminator.Crash("Runtime trampolines are not supported on this "
                   "architecture. Recompile without -fsafe-trampoline "
                   "to use the legacy stack-trampoline path.");
#endif
}

void *RTDEF(TrampolineInit)(
    void *scratch, const void *calleeAddress, const void *staticChainAddress) {
  crashIfUnsupported();
  auto &pool{trampoline::TrampolinePool::instance()};
  return pool.allocate(calleeAddress, staticChainAddress);
}

void *RTDEF(TrampolineAdjust)(void *handle) {
  crashIfUnsupported();
  auto &pool{trampoline::TrampolinePool::instance()};
  return pool.getCallableAddress(
      static_cast<trampoline::TrampolineHandle *>(handle));
}

void RTDEF(TrampolineFree)(void *handle) {
  crashIfUnsupported();
  auto &pool{trampoline::TrampolinePool::instance()};
  pool.free(static_cast<trampoline::TrampolineHandle *>(handle));
}

} // extern "C"
} // namespace Fortran::runtime
