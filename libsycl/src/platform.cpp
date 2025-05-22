//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/platform.hpp>

#include <stdexcept>

__SYCL_BEGIN_VERSIONED_NAMESPACE

platform::platform() { throw std::runtime_error("Unimplemented"); }

__SYCL_END_VERSIONED_NAMESPACE
