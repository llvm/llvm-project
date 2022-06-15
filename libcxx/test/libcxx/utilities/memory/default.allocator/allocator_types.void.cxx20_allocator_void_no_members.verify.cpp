//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that members of std::allocator<void> are not provided in C++20
// with _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_VOID_SPECIALIZATION but without
// _LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_MEMBERS.

// UNSUPPORTED: c++03, c++11, c++14, c++17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_CXX20_REMOVED_ALLOCATOR_VOID_SPECIALIZATION
//
// Ignore any extra errors arising from typo correction.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

#include <memory>

std::allocator<void>::pointer x;            // expected-error-re {{no {{(type|template)}} named 'pointer'}}
std::allocator<void>::const_pointer y;      // expected-error-re {{no {{(type|template)}} named 'const_pointer'}}
std::allocator<void>::value_type z;         // expected-error-re {{no {{(type|template)}} named 'value_type'}}
std::allocator<void>::rebind<int>::other t; // expected-error-re {{no {{(type|template)}} named 'rebind'}}
