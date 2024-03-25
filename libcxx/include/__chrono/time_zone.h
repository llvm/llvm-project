// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP___CHRONO_TIME_ZONE_H
#define _LIBCPP___CHRONO_TIME_ZONE_H

#include <version>
// Enable the contents of the header only when libc++ was built with experimental features enabled.
#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#  include <__compare/strong_order.h>
#  include <__config>
#  include <__memory/unique_ptr.h>
#  include <string_view>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

_LIBCPP_PUSH_MACROS
#  include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#  if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM) &&   \
      !defined(_LIBCPP_HAS_NO_LOCALIZATION)

namespace chrono {

class _LIBCPP_AVAILABILITY_TZDB time_zone {
  _LIBCPP_HIDE_FROM_ABI time_zone() = default;

public:
  class __impl; // public so it can be used by make_unique.

  // The "constructor".
  //
  // The default constructor is private to avoid the constructor from being
  // part of the ABI. Instead use an __ugly_named function as an ABI interface,
  // since that gives us the ability to change it in the future.
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI static time_zone __create(unique_ptr<__impl>&& __p);

  _LIBCPP_EXPORTED_FROM_ABI ~time_zone();

  _LIBCPP_HIDE_FROM_ABI time_zone(time_zone&&)            = default;
  _LIBCPP_HIDE_FROM_ABI time_zone& operator=(time_zone&&) = default;

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI string_view name() const noexcept { return __name(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const __impl& __implementation() const noexcept { return *__impl_; }

private:
  [[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string_view __name() const noexcept;
  unique_ptr<__impl> __impl_;
};

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_HIDE_FROM_ABI inline bool
operator==(const time_zone& __x, const time_zone& __y) noexcept {
  return __x.name() == __y.name();
}

_LIBCPP_NODISCARD_EXT _LIBCPP_AVAILABILITY_TZDB _LIBCPP_HIDE_FROM_ABI inline strong_ordering
operator<=>(const time_zone& __x, const time_zone& __y) noexcept {
  return __x.name() <=> __y.name();
}

} // namespace chrono

#  endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_TIME_ZONE_DATABASE) && !defined(_LIBCPP_HAS_NO_FILESYSTEM)
         // && !defined(_LIBCPP_HAS_NO_LOCALIZATION)

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_TZDB)

#endif // _LIBCPP___CHRONO_TIME_ZONE_H
