//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <type_traits>

// is_trivial and is_trivial_v are deprecated in C++26 by P3247R2

#include <type_traits>

static_assert(std::is_trivial<int>::value);
// expected-warning@-1 {{'is_trivial<int>' is deprecated: Consider using is_trivially_copyable<T>::value && is_trivially_default_constructible<T>::value instead.}}
static_assert(std::is_trivial_v<int>);
// expected-warning@-1 {{'is_trivial_v<int>' is deprecated: Consider using is_trivially_copyable_v<T> && is_trivially_default_constructible_v<T> instead.}}
