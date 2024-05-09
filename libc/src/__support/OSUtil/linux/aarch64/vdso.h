//===---------- aarch64 vdso configuration ------------------------* C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "src/__support/CPP/string_view.h"
namespace LIBC_NAMESPACE {
namespace vdso {
// macro definitions
#define LIBC_VDSO_HAS_RT_SIGRETURN
#define LIBC_VDSO_HAS_GETTIMEOFDAY
#define LIBC_VDSO_HAS_CLOCK_GETTIME
#define LIBC_VDSO_HAS_CLOCK_GETRES

// list of VDSO symbols
enum class VDSOSym {
  RTSigReturn,
  GetTimeOfDay,
  ClockGetTime,
  ClockGetRes,
  VDSOSymCount
};

// translate VDSOSym to symbol names
LIBC_INLINE constexpr cpp::string_view symbol_name(VDSOSym sym) {
  switch (sym) {
  case VDSOSym::RTSigReturn:
    return "__kernel_rt_sigreturn";
  case VDSOSym::GetTimeOfDay:
    return "__kernel_gettimeofday";
  case VDSOSym::ClockGetTime:
    return "__kernel_clock_gettime";
  case VDSOSym::ClockGetRes:
    return "__kernel_clock_getres";
  default:
    return "";
  }
}

// symbol versions
LIBC_INLINE constexpr cpp::string_view symbol_version(VDSOSym) {
  return "LINUX_2.6.39";
}
} // namespace vdso
} // namespace LIBC_NAMESPACE
