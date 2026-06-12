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

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// This helper is not included to LiboffloadMock to keep LiboffloadMock isolated
// from libsycl implementation and to not introduce extra operations for tests
// where devices enumeration logic is not important. LiboffloadMock provides
// default single gpu-device configuration that is enough for most of the tests.
// For tests where devices enumeration logic is important, UnittestsHelper
// allows to call global state reset and platforms initialization methods to be
// able to set expectations on devices enumeration calls in a proper way.
class UnittestsHelper {
public:
  static void initPlatforms() { GlobalHandler::initPlatforms(); }

  static void resetGlobalObjects() { GlobalHandler::resetGlobalObjects(); }
};

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UNITTESTS_COMMON_UNITTESTS_HELPER_HPP
