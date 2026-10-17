//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-inplace-vector

#include <cstddef>
#include <inplace_vector>
#include <limits>

template <class T, std::size_t _Count, class SizeType>
struct InplaceVectorLayout {
  T foo[_Count];
  SizeType bar;
};

static_assert(sizeof(std::inplace_vector<char, std::numeric_limits<unsigned char>::max() - 1>) ==
              sizeof(InplaceVectorLayout<char, std::numeric_limits<unsigned char>::max() - 1, unsigned char>));
static_assert(sizeof(std::inplace_vector<char, std::numeric_limits<unsigned short>::max() - 1>) ==
              sizeof(InplaceVectorLayout<char, std::numeric_limits<unsigned short>::max() - 1, unsigned short>));
static_assert(sizeof(std::inplace_vector<char, std::numeric_limits<unsigned int>::max() - 1>) ==
              sizeof(InplaceVectorLayout<char, std::numeric_limits<unsigned int>::max() - 1, unsigned int>));
// Divide by 64 to not run into max structure size issues
static_assert(sizeof(std::inplace_vector<char, (std::numeric_limits<unsigned long>::max() / 64) - 1>) ==
              sizeof(InplaceVectorLayout<char, (std::numeric_limits<unsigned long>::max() / 64) - 1, unsigned long>));
static_assert(sizeof(std::inplace_vector<char, (std::numeric_limits<std::size_t>::max() / 64) - 1>) ==
              sizeof(InplaceVectorLayout<char, (std::numeric_limits<std::size_t>::max() / 64) - 1, std::size_t>));
