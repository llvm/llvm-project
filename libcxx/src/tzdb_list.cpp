//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// For information see https://libcxx.llvm.org/DesignDocs/TimeZone.html

#include <chrono>

#include "include/tzdb/tzdb_list_private.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace chrono {

_LIBCPP_EXPORTED_FROM_ABI tzdb_list::~tzdb_list() { delete __impl_; }

_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI const tzdb& tzdb_list::front() const noexcept {
  return __impl_->front();
}

_LIBCPP_EXPORTED_FROM_ABI tzdb_list::const_iterator tzdb_list::erase_after(const_iterator __p) {
  return __impl_->erase_after(__p);
}

_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI tzdb_list::const_iterator tzdb_list::begin() const noexcept {
  return __impl_->begin();
}
_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI tzdb_list::const_iterator tzdb_list::end() const noexcept {
  return __impl_->end();
}

_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI tzdb_list::const_iterator tzdb_list::cbegin() const noexcept {
  return __impl_->cbegin();
}
_LIBCPP_NODISCARD_EXT _LIBCPP_EXPORTED_FROM_ABI tzdb_list::const_iterator tzdb_list::cend() const noexcept {
  return __impl_->cend();
}

} // namespace chrono

_LIBCPP_END_NAMESPACE_STD
