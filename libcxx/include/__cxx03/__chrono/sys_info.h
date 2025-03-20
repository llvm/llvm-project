// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CXX03___CHRONO_SYS_INFO_H
#define _LIBCPP___CXX03___CHRONO_SYS_INFO_H

#include <__cxx03/version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#  include <__cxx03/__chrono/duration.h>
#  include <__cxx03/__chrono/system_clock.h>
#  include <__cxx03/__chrono/time_point.h>
#  include <__cxx03/__config>
#  include <__cxx03/string>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20

namespace chrono {

struct sys_info {
  sys_seconds begin;
  sys_seconds end;
  seconds offset;
  minutes save;
  string abbrev;
};

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#endif // _LIBCPP___CXX03___CHRONO_SYS_INFO_H
