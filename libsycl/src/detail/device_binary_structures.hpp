//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of device image types and structures
/// used for offloading.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_BINARY_STRUCTURES
#define _LIBSYCL_DEVICE_BINARY_STRUCTURES

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// Target identification strings.
///
/// A device type represented by a particular target
/// triple requires specific binary images. We need
/// to map the image type onto the device target triple.

/// SPIR-V with 64-bit pointers.
static constexpr char DeviceBinaryTripleSPIRV64[] = "spirv64-unknown-unknown";

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_BINARY_STRUCTURES
