//==---------------- platform.cpp - SYCL platform --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/platform.hpp>

#include <stdexcept>

namespace sycl {
inline namespace _V1 {
platform::platform() { throw std::runtime_error("Unimplemented"); }
} // namespace _V1
} // namespace sycl
