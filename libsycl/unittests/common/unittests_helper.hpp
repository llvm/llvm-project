//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper utilities for libsycl unit tests.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_UNITTESTS_COMMON_UNITTESTS_HELPER_HPP
#define _LIBSYCL_UNITTESTS_COMMON_UNITTESTS_HELPER_HPP

#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>
#include <mock/helpers.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace unittests {

// This helper is not included to LiboffloadMock to keep LiboffloadMock isolated
// from libsycl implementation and to not introduce extra operations for tests
// where devices enumeration logic is not important. LiboffloadMock provides
// default single gpu-device configuration that is enough for most of the tests.
// For tests where devices enumeration logic is important, UnittestsHelper
// allows to call global state reset and platforms initialization methods to be
// able to set expectations on devices enumeration calls in a proper way.
struct UnittestsHelper {
  UnittestsHelper() { detail::PlatformImpl::rediscoverIfEmpty = true; }

  ~UnittestsHelper() {
    if (!detail::getPlatformCache().empty()) {
      detail::getPlatformCache().clear();
      detail::getOffloadTopologies() = {};
    }
  }

  mock::MockWrapper Mock;
};

} // namespace unittests
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UNITTESTS_COMMON_UNITTESTS_HELPER_HPP
