//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host report path API interface.
///
//===----------------------------------------------------------------------===//

#include "dasan.h"
#include "dasan_report.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __sanitizer;

template <bool Abort>
static bool Report(uintptr_t Addr, uint64_t AccessSize, bool IsWrite,
                   uintptr_t Pc, uintptr_t Bp, uint64_t Base, uint64_t Size) {
  return __dasan::HandleHostReport(Addr, AccessSize, Base, Size, IsWrite,
                                   !Abort, Pc, Bp);
}

// The return address is the faulting access, so these must stay real calls.
#define DASAN_HANDLER(Name, IsWrite)                                          \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE                                    \
      [[gnu::cold, gnu::noinline, gnu::noreturn]] void                        \
      __dasan_report_##Name(void* Addr, uint64_t AccessSize, uint64_t Base,   \
                            uint64_t Size) {                                  \
    __dasan::Initialize();                                                    \
    GET_CALLER_PC_BP;                                                         \
    Report<true>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite, pc,  \
                 bp, Base, Size);                                             \
    Die();                                                                    \
  }                                                                           \
  extern "C" SANITIZER_INTERFACE_ATTRIBUTE [[gnu::cold, gnu::noinline]] void  \
  __dasan_report_##Name##_noabort(void* Addr, uint64_t AccessSize,            \
                                  uint64_t Base, uint64_t Size) {             \
    __dasan::Initialize();                                                    \
    GET_CALLER_PC_BP;                                                         \
    if (Report<false>(reinterpret_cast<uintptr_t>(Addr), AccessSize, IsWrite, \
                      pc, bp, Base, Size))                                    \
      Die();                                                                  \
  }

DASAN_HANDLER(load, false)
DASAN_HANDLER(store, true)

#undef DASAN_HANDLER
