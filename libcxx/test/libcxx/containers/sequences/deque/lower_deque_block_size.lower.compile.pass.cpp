//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// Test that std::__deque_block_size has lowered sizes when _LIBCPP_ABI_USE_LOWER_DEQUE_BLOCK_SIZE is defined to 1.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_USE_LOWER_DEQUE_BLOCK_SIZE=1

#include <deque>
#include <cstddef>

template <std::size_t Size>
struct TypeOfSize {
  char data[Size];
};

static_assert(_LIBCPP_ABI_USE_LOWER_DEQUE_BLOCK_SIZE == 1, "");

static_assert(std::__deque_block_size<char, std::ptrdiff_t>::value == 512, "");
static_assert(std::__deque_block_size<int, std::ptrdiff_t>::value == 128, "");
static_assert(std::__deque_block_size<double, std::ptrdiff_t>::value == 64, "");

static_assert(std::__deque_block_size<TypeOfSize<255>, std::ptrdiff_t>::value == 2, "");
static_assert(std::__deque_block_size<TypeOfSize<256>, std::ptrdiff_t>::value == 2, "");
static_assert(std::__deque_block_size<TypeOfSize<512>, std::ptrdiff_t>::value == 2, "");

int main(int, char**) { return 0; }
