// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_TZDB_H
#define _LIBCPP___CHRONO_TZDB_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#  include <__chrono/leap_second.h>
#  include <__chrono/time_zone.h>
#  include <__chrono/time_zone_link.h>
#  include <__config>
#  include <string>
#  include <vector>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_PUSH_MACROS
#  include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono {

struct tzdb {
  string version;
  vector<time_zone> zones;
  vector<time_zone_link> links;

  vector<leap_second> leap_seconds;
};

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#endif // _LIBCPP___CHRONO_TZDB_H
