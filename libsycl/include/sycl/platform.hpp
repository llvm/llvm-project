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

#ifndef __LIBSYCL_PLATFORM_HPP
#define __LIBSYCL_PLATFORM_HPP

#include <sycl/detail/config.hpp>

__SYCL_BEGIN_VERSIONED_NAMESPACE

class __SYCL_EXPORT platform {
public:
  /// Constructs a SYCL platform using the default device.
  platform();

}; // class platform

__SYCL_END_VERSIONED_NAMESPACE

#endif // __LIBSYCL_PLATFORM_HPP
