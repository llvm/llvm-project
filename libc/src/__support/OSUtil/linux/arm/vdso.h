//===---------- arm vdso configuration ----------------------------* C++ *-===//
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
#define LIBC_VDSO_HAS_GETTIMEOFDAY
#define LIBC_VDSO_HAS_CLOCK_GETTIME

// list of VDSO symbols
enum class VDSOSym {
  GetTimeOfDay,
  ClockGetTime,
};

// translate VDSOSym to symbol names
LIBC_INLINE constexpr cpp::string_view symbol_name(VDSOSym sym) {
  switch (sym) {
  case VDSOSym::GetTimeOfDay:
    return "__vdso_gettimeofday";
  case VDSOSym::ClockGetTime:
    return "__vdso_clock_gettime";
  }
}

// symbol versions
LIBC_INLINE constexpr cpp::string_view symbol_version(VDSOSym) {
  return "LINUX_2.6";
}
} // namespace vdso
} // namespace LIBC_NAMESPACE
