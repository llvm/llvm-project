//==---------------- platform.hpp - SYCL platform --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBSYCL_PLATFORM_HPP
#define __LIBSYCL_PLATFORM_HPP

#include <sycl/detail/export.hpp>

namespace sycl {
inline namespace _V1 {
/// Encapsulates a SYCL platform on which kernels may be executed.
///
/// \ingroup sycl_api
class __SYCL_EXPORT platform {
public:
  /// Constructs a SYCL platform using the default device.
  platform();

}; // class platform
} // namespace _V1
} // namespace sycl

#endif // __LIBSYCL_PLATFORM_HPP