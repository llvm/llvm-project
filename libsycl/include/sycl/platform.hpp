//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the Platform class, which encapsulates
/// a single SYCL platform on which kernel functions may be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_PLATFORM_HPP
#define _LIBSYCL_PLATFORM_HPP

#include <sycl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class _LIBSYCL_EXPORT platform {
public:
  /// Constructs a SYCL platform using the default device.
  platform();

}; // class platform

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_PLATFORM_HPP
