//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL platform class, which
/// encapsulates a single platform on which kernel functions may be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___PLATFORM_HPP
#define _LIBSYCL___PLATFORM_HPP

#include <sycl/__detail/__config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class _LIBSYCL_EXPORT platform {
public:
  /// Constructs a SYCL platform which contains the default device.
  platform();

}; // class platform

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___PLATFORM_HPP
