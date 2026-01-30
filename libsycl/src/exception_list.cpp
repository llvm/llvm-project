//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// SYCL 2020 4.13.2. Exception class interface.
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/exception.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

exception_list::size_type exception_list::size() const { return MList.size(); }

exception_list::iterator exception_list::begin() const { return MList.begin(); }

exception_list::iterator exception_list::end() const { return MList.cend(); }

_LIBSYCL_END_NAMESPACE_SYCL
