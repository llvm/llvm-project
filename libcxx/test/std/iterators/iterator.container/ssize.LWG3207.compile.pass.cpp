//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>

// template <class T, ptrdiff_t N>
//   constexpr ptrdiff_t ssize(const T (&array)[N]) noexcept;
//
// This test checks that the library implements LWG3207 as not-a-defect.
// `clang -m32` is an example of a configuration where using ptrdiff_t
// instead of size_t in std::ssize has an observable SFINAE effect.
//
// REQUIRES: 32-bit-pointer

#include <iterator>
#include <climits>
#include <cstddef>

// Test the test:
static_assert(sizeof(std::ptrdiff_t) == 4, "Run only on these platforms");
static_assert(sizeof(std::size_t) == 4, "Run only on these platforms");
static_assert(std::size_t(PTRDIFF_MAX) + 1 > std::size_t(PTRDIFF_MAX), "This should always be true");
extern char forming_this_type_must_be_valid_on_this_platform[std::size_t(PTRDIFF_MAX) + 1];

// The actual test:
template <class T>
concept HasSsize = requires(T&& t) { std::ssize(t); };

static_assert(HasSsize<char[std::size_t(PTRDIFF_MAX)]>);
static_assert(!HasSsize<char[std::size_t(PTRDIFF_MAX) + 1]>);
