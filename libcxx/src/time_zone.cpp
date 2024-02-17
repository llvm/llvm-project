//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#include <chrono>

#include "include/tzdb/time_zone_private.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

[[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI time_zone time_zone::__create(unique_ptr<time_zone::__impl>&& __p) {
  _LIBCPP_ASSERT_NON_NULL(__p != nullptr, "initialized time_zone without a valid pimpl object");
  time_zone result;
  result.__impl_ = std::move(__p);
  return result;
}

_LIBCPP_EXPORTED_FROM_ABI time_zone::~time_zone() = default;

[[nodiscard]] _LIBCPP_EXPORTED_FROM_ABI string_view time_zone::__name() const noexcept { return __impl_->__name(); }

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD
