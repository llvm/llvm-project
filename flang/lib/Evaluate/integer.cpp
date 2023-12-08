//===-- lib/Evaluate/integer.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/integer.h"

namespace Fortran::evaluate::value {

template class Integer<8>;
template class Integer<16>;
template class Integer<32>;
template class Integer<64>;
template class Integer<80>;
template class Integer<128>;

// Sanity checks against misconfiguration bugs
static_assert(Integer<8>::partBits == 8);
static_assert(std::is_same_v<typename Integer<8>::Part, std::uint8_t>);
static_assert(Integer<16>::partBits == 16);
static_assert(std::is_same_v<typename Integer<16>::Part, std::uint16_t>);
static_assert(Integer<32>::partBits == 32);
static_assert(std::is_same_v<typename Integer<32>::Part, std::uint32_t>);
static_assert(Integer<64>::partBits == 32);
static_assert(std::is_same_v<typename Integer<64>::Part, std::uint32_t>);
static_assert(Integer<128>::partBits == 32);
static_assert(std::is_same_v<typename Integer<128>::Part, std::uint32_t>);
} // namespace Fortran::evaluate::value
