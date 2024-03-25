//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <functional>

// template<class R, class F, class... Args>
// constexpr R invoke_r(F&& f, Args&&... args)              // C++23
//     noexcept(is_nothrow_invocable_r_v<R, F, Args...>);
//
// Make sure that we diagnose when std::invoke_r is used with a return type that
// would yield a dangling reference to a temporary.

// TODO: We currently can't diagnose because we don't implement reference_converts_from_temporary.
// XFAIL: *

#include <functional>
#include <cassert>

#include "test_macros.h"

void f() {
    auto func = []() -> int { return 0; };
    std::invoke_r<int&&>(func); // expected-error {{Returning from invoke_r would bind a temporary object}}
}
