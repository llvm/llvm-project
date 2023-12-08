//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// const char* what() const noexcept override;

#include <expected>
#include <utility>

template <class T>
concept WhatNoexcept =
    requires(const T& t) {
      { t.what() } noexcept;
    };

struct foo{};

static_assert(!WhatNoexcept<foo>);
static_assert(WhatNoexcept<std::bad_expected_access<int>>);
static_assert(WhatNoexcept<std::bad_expected_access<foo>>);
