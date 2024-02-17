// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#ifndef _LIBCPP_SRC_INCLUDE_TZDB_TIME_ZONE_PRIVATE_H
#define _LIBCPP_SRC_INCLUDE_TZDB_TIME_ZONE_PRIVATE_H

#include <chrono>
#include <string>
#include <vector>

#include "types_private.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

class time_zone::__impl {
public:
  explicit _LIBCPP_HIDE_FROM_ABI __impl(string&& __name) : __name_(std::move(__name)) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI string_view __name() const noexcept { return __name_; }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI vector<__tz::__continuation>& __continuations() { return __continuations_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI const vector<__tz::__continuation>& __continuations() const {
    return __continuations_;
  }

private:
  string __name_;
  // Note the first line has a name + __continuation, the other lines
  // are just __continuations. So there is always at least one item in
  // the vector.
  vector<__tz::__continuation> __continuations_;
};

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_SRC_INCLUDE_TZDB_TIME_ZONE_PRIVATE_H
