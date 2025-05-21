//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <future>

// template<class R, class... ArgTypes>
//     class packaged_task<R(ArgTypes...)>
// {
// public:
//     typedef R result_type; // extension

// This libc++ extension is deprecated. See https://github.com/llvm/llvm-project/issues/112856.

#include <future>
#include <type_traits>

struct A {};

using RA = std::packaged_task<A(int, char)>::result_type;    // expected-warning {{'result_type' is deprecated}}
using RV = std::packaged_task<void(int, char)>::result_type; // expected-warning {{'result_type' is deprecated}}
