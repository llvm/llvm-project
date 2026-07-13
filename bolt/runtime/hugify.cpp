//===- bolt/runtime/hugify.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#if (defined(__x86_64__) || defined(__aarch64__) || defined(__arm64__)) &&     \
    !defined(__APPLE__)

#include "common.h"

#pragma GCC visibility push(hidden)

// Enables a very verbose logging to stderr useful when debugging
// #define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
#define DEBUG(X)                                                               \
  { X; }
#else
#define DEBUG(X)                                                               \
  {}
#endif

// Function constrains trampoline to _start,
// so we can resume regular execution of the function that we hooked.
extern void __bolt_hugify_start_program();

// The __hot_start and __hot_end symbols set by Bolt. We use them to figure
// out the rage for marking huge pages.
extern uint64_t __hot_start;
extern uint64_t __hot_end;

static void getKernelVersion(uint32_t *Val) {
  // release should be in the format: %d.%d.%d
  // major, minor, release
  struct UtsNameTy UtsName;
  int Ret = __uname(&UtsName);
  const char *Buf = UtsName.release;
  const char *End = Buf + strLen(Buf);
  const char Delims[2][2] = {".", "."};

  for (int i = 0; i < 3; ++i) {
    if (!scanUInt32(Buf, End, Val[i])) {
      return;
    }
    if (i < sizeof(Delims) / sizeof(Delims[0])) {
      const char *Ptr = Delims[i];
      while (*Ptr != '\0') {
        if (*Ptr != *Buf) {
          return;
        }
        ++Ptr;
        ++Buf;
      }
    }
  }
}

/// Check whether the kernel supports THP via corresponding sysfs entry.
/// thp works only starting from 5.10
static bool hasPagecacheTHPSupport() {
  char Buf[64];

  int FD = __open("/sys/kernel/mm/transparent_hugepage/enabled",
                  0 /* O_RDONLY */, 0);
  if (FD < 0)
    return false;

  memset(Buf, 0, sizeof(Buf));
  const size_t Res = __read(FD, Buf, sizeof(Buf));
  if (Res < 0)
    return false;

  if (!strStr(Buf, "[always]") && !strStr(Buf, "[madvise]")) {
    DEBUG(report("[hugify] THP support is not enabled.\n");)
    return false;
  }

  struct KernelVersionTy {
    uint32_t major;
    uint32_t minor;
    uint32_t release;
  };

  KernelVersionTy KernelVersion;

  getKernelVersion((uint32_t *)&KernelVersion);
  if (KernelVersion.major >= 6 ||
      (KernelVersion.major == 5 && KernelVersion.minor >= 10))
    return true;

  return false;
}

static void hugifyForOldKernel(uint8_t *From, uint8_t *To) {
  const size_t Size = To - From;

  uint8_t *Mem = reinterpret_cast<uint8_t *>(
      __mmap(0, Size, 0x3 /* PROT_READ | PROT_WRITE */,
             0x22 /* MAP_PRIVATE | MAP_ANONYMOUS */, -1, 0));

  if (Mem == ((void *)-1) /* MAP_FAILED */) {
    char Msg[] = "[hugify] could not allocate memory for text move\n";
    reportError(Msg, sizeof(Msg));
  }

  DEBUG(reportNumber("[hugify] allocated temporary address: ", (uint64_t)Mem,
                     16);)
  DEBUG(reportNumber("[hugify] allocated size: ", (uint64_t)Size, 16);)

  // Copy the hot code to a temporary location.
  memcpy(Mem, From, Size);

  __prctl(41 /* PR_SET_THP_DISABLE */, 0, 0, 0, 0);
  // Maps out the existing hot code.
  if (__mmap(reinterpret_cast<uint64_t>(From), Size,
             0x3 /* PROT_READ | PROT_WRITE */,
             0x32 /* MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE */, -1,
             0) == ((void *)-1) /*MAP_FAILED*/) {
    char Msg[] =
        "[hugify] failed to mmap memory for large page move terminating\n";
    reportError(Msg, sizeof(Msg));
  }

  // Mark the hot code page to be huge page.
  if (__madvise(From, Size, 14 /* MADV_HUGEPAGE */) == -1) {
    char Msg[] = "[hugify] setting MADV_HUGEPAGE is failed\n";
    reportError(Msg, sizeof(Msg));
  }

  // Copy the hot code back.
  memcpy(From, Mem, Size);

  // Change permission back to read-only, ignore failure
  __mprotect(From, Size, 0x5 /* PROT_READ | PROT_EXEC */);

  __munmap(Mem, Size);
}

extern "C" void __bolt_hugify_self_impl() {
  uint8_t *HotStart = (uint8_t *)&__hot_start;
  uint8_t *HotEnd = (uint8_t *)&__hot_end;
  // Make sure the start and end are aligned with huge page address
  const size_t HugePageBytes = 2L * 1024 * 1024;
  uint8_t *From = HotStart - ((intptr_t)HotStart & (HugePageBytes - 1));
  uint8_t *To = HotEnd + (HugePageBytes - 1);
  To -= (intptr_t)To & (HugePageBytes - 1);

  DEBUG(reportNumber("[hugify] hot start: ", (uint64_t)HotStart, 16);)
  DEBUG(reportNumber("[hugify] hot end: ", (uint64_t)HotEnd, 16);)
  DEBUG(reportNumber("[hugify] aligned huge page from: ", (uint64_t)From, 16);)
  DEBUG(reportNumber("[hugify] aligned huge page to: ", (uint64_t)To, 16);)

  if (!hasPagecacheTHPSupport()) {
    DEBUG(report(
              "[hugify] workaround with memory alignment for kernel < 5.10\n");)
    hugifyForOldKernel(From, To);
    return;
  }

  if (__madvise(From, (To - From), 14 /* MADV_HUGEPAGE */) == -1) {
    char Msg[] = "[hugify] failed to allocate large page\n";
    // TODO: allow user to control the failure behavior.
    reportError(Msg, sizeof(Msg));
  }
}

/// This is hooking ELF's entry, it needs to save all machine state.
extern "C" __attribute((naked)) void __bolt_hugify_self() {
  // clang-format off
#if defined(__x86_64__)
  __asm__ __volatile__(SAVE_ALL "call __bolt_hugify_self_impl\n" RESTORE_ALL
                                "jmp __bolt_hugify_start_program\n"
                                :::);
#elif defined(__aarch64__) || defined(__arm64__)
  __asm__ __volatile__(SAVE_ALL "bl __bolt_hugify_self_impl\n" RESTORE_ALL
                                "adrp x16, __bolt_hugify_start_program\n"
                                "add x16, x16, #:lo12:__bolt_hugify_start_program\n"
                                "br x16\n"
                                :::);
#else
  __exit(1);
#endif
  // clang-format on
}
#endif
