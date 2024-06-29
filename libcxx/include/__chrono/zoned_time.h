// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_ZONED_TIME_H
#define _LIBCPP___CHRONO_ZONED_TIME_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#  include <__chrono/time_zone.h>
#  include <__chrono/tzdb_list.h>
#  include <__config>
#  include <__fwd/string_view.h>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono {

template <class>
struct zoned_traits {};

template <>
struct zoned_traits<const time_zone*> {
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static const time_zone* default_zone() { return chrono::locate_zone("UTC"); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static const time_zone* locate_zone(string_view __name) {
    return chrono::locate_zone(__name);
  }
};

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_EXPERIMENTAL_TZDB)

#endif // _LIBCPP___CHRONO_ZONED_TIME_H
