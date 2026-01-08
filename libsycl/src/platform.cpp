//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/platform.hpp>

#include <stdexcept>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

platform::platform() { throw std::runtime_error("Unimplemented"); }

_LIBSYCL_END_NAMESPACE_SYCL
