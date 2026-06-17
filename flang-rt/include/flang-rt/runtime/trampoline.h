//===-- flang-rt/runtime/trampoline.h ----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Internal declarations for the W^X-compliant trampoline pool.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_TRAMPOLINE_H_
#define FLANG_RT_RUNTIME_TRAMPOLINE_H_

#include <cstddef>
#include <cstdint>

namespace Fortran::runtime::trampoline {

/// Per-trampoline data entry. Stored in a writable (non-executable) region.
/// Each entry is paired with a trampoline code stub in the executable region.
struct TrampolineData {
  const void *calleeAddress{nullptr};
  const void *staticChainAddress{nullptr};
};

/// Default number of trampoline slots in the pool.
/// Can be overridden via FLANG_TRAMPOLINE_POOL_SIZE environment variable.
constexpr std::size_t kDefaultPoolSize{1024};

/// Size of each trampoline code stub in bytes (platform-specific).
#if defined(__x86_64__) || defined(_M_X64)
// x86-64 trampoline stub:
//   movq TDATA_OFFSET(%rip), %r10    # load static chain from TDATA
//   movabsq $0, %r11                 # placeholder for callee address
//   jmpq *%r11
// Actually we use an indirect approach through the TDATA pointer:
//   movq (%r10), %r10                # load static chain (8 bytes)
//   -- but we need the TDATA pointer first
// Simplified approach for x86-64:
//   leaq tdata_entry(%rip), %r11     # get TDATA entry address
//   movq 8(%r11), %r10               # load static chain
//   jmpq *(%r11)                     # jump to callee
constexpr std::size_t kTrampolineStubSize{32};
constexpr int kNestRegister{10}; // %r10 is the nest/static chain register
#elif defined(__aarch64__) || defined(_M_ARM64)
// AArch64 trampoline stub:
//   adr x17, tdata_entry             # get TDATA entry address
//   ldr x15, [x17, #8]              # load static chain into x15 (nest reg)
//   ldr x17, [x17]                  # load callee address
//   br x17
constexpr std::size_t kTrampolineStubSize{32};
constexpr int kNestRegister{15}; // x15 is the nest / static-chain register
#elif defined(__powerpc64__) || defined(__ppc64__)
constexpr std::size_t kTrampolineStubSize{48};
constexpr int kNestRegister{11}; // r11
#else
// Fallback: generous size
constexpr std::size_t kTrampolineStubSize{64};
constexpr int kNestRegister{0};
#endif

} // namespace Fortran::runtime::trampoline

#endif // FLANG_RT_RUNTIME_TRAMPOLINE_H_
