//===---------- x86/x86_64 vdso configuration ---------------------* C++ *-===//
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
#define LIBC_VDSO_HAS_CLOCK_GETTIME
#define LIBC_VDSO_HAS_GETCPU
#define LIBC_VDSO_HAS_GETTIMEOFDAY
#define LIBC_VDSO_HAS_TIME

// list of VDSO symbols
enum class VDSOSym { ClockGetTime, GetCpu, GetTimeOfDay, Time, VDSOSymCount };

// translate VDSOSym to symbol names
LIBC_INLINE constexpr cpp::string_view symbol_name(VDSOSym sym) {
  switch (sym) {
  case VDSOSym::ClockGetTime:
    return "__vdso_clock_gettime";
  case VDSOSym::GetCpu:
    return "__vdso_getcpu";
  case VDSOSym::GetTimeOfDay:
    return "__vdso_gettimeofday";
  case VDSOSym::Time:
    return "__vdso_time";
  default:
    return "";
  }
}

// symbol versions
LIBC_INLINE constexpr cpp::string_view symbol_version(VDSOSym) {
  return "LINUX_2.6";
}
} // namespace vdso
} // namespace LIBC_NAMESPACE
