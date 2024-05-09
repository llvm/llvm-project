//===---------- RISC-V vdso configuration -------------------------* C++ *-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_RISCV_VDSO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_RISCV_VDSO_H
#include "src/__support/CPP/string_view.h"
namespace LIBC_NAMESPACE {
namespace vdso {
// macro definitions
#define LIBC_VDSO_HAS_RT_SIGRETURN
#define LIBC_VDSO_HAS_GETTIMEOFDAY
#define LIBC_VDSO_HAS_CLOCK_GETTIME
#define LIBC_VDSO_HAS_CLOCK_GETRES
#define LIBC_VDSO_HAS_GETCPU
#define LIBC_VDSO_HAS_FLUSH_ICACHE

// list of VDSO symbols
enum class VDSOSym {
  RTSigReturn,
  GetTimeOfDay,
  ClockGetTime,
  ClockGetRes,
  GetCpu,
  FlushICache,
  VDSOSymCount
};

// translate VDSOSym to symbol names
LIBC_INLINE constexpr cpp::string_view symbol_name(VDSOSym sym) {
  switch (sym) {
  case VDSOSym::RTSigReturn:
    return "__vdso_rt_sigreturn";
  case VDSOSym::GetTimeOfDay:
    return "__vdso_gettimeofday";
  case VDSOSym::ClockGetTime:
    return "__vdso_clock_gettime";
  case VDSOSym::ClockGetRes:
    return "__vdso_clock_getres";
  case VDSOSym::GetCpu:
    return "__vdso_getcpu";
  case VDSOSym::FlushICache:
    return "__vdso_flush_icache";
  default:
    return "";
  }
}

// symbol versions
LIBC_INLINE constexpr cpp::string_view symbol_version(VDSOSym) {
  return "LINUX_4.15";
}
} // namespace vdso
} // namespace LIBC_NAMESPACE
#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_RISCV_VDSO_H
