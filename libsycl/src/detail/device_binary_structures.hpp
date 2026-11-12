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

#ifndef _LIBSYCL_SRC_DETAIL_DEVICE_BINARY_STRUCTURES_HPP
#define _LIBSYCL_SRC_DETAIL_DEVICE_BINARY_STRUCTURES_HPP

#include <sycl/__impl/detail/config.hpp>

#include <string_view>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

/// Target identification strings.
///
/// A device type represented by a particular target
/// triple requires specific binary images. We need
/// to map the image type onto the device target triple.
/// Offload images record the triple in its normalized three-part form, of which
/// the architecture component alone identifies the binary format.

/// SPIR-V with 64-bit pointers.
static constexpr std::string_view DeviceBinaryArchSPIRV64 = "spirv64";

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SRC_DETAIL_DEVICE_BINARY_STRUCTURES_HPP
