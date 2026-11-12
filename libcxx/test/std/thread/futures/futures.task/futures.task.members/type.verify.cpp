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

// Verify that the non-conforming extension packaged_task::result_type is removed.
// See https://llvm.org/PR112856.

#include <future>

struct A {};

using RA = std::packaged_task<A(int, char)>::result_type;    // expected-error {{no type named 'result_type'}}
using RV = std::packaged_task<void(int, char)>::result_type; // expected-error {{no type named 'result_type'}}
