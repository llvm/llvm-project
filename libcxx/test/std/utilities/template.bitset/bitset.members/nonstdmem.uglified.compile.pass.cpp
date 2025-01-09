//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <bitset>

// This test ensures that we don't use a non-uglified name 'base', 'iterator',
// 'const_iterator', and `const_reference` in the implementation of bitset.
//
// See https://github.com/llvm/llvm-project/issues/111125.
// See https://github.com/llvm/llvm-project/issues/121618.

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <cstddef>
#include <bitset>
#include <type_traits>

struct my_base {
  typedef int* iterator;
  typedef const int* const_iterator;
  typedef my_base base;
  typedef const int& const_reference;
};

template <std::size_t N>
struct my_derived : my_base, std::bitset<N> {};

static_assert(std::is_same<my_derived<0>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<1>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<8>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<12>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<16>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<32>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<48>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<64>::iterator, int*>::value, "");
static_assert(std::is_same<my_derived<96>::iterator, int*>::value, "");

static_assert(std::is_same<my_derived<0>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<1>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<8>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<12>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<16>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<32>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<48>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<64>::const_iterator, const int*>::value, "");
static_assert(std::is_same<my_derived<96>::const_iterator, const int*>::value, "");

static_assert(std::is_same<my_derived<0>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<1>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<8>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<12>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<16>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<32>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<48>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<64>::base, my_base>::value, "");
static_assert(std::is_same<my_derived<96>::base, my_base>::value, "");

static_assert(std::is_same<my_derived<0>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<1>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<8>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<12>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<16>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<32>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<48>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<64>::const_reference, const int&>::value, "");
static_assert(std::is_same<my_derived<96>::const_reference, const int&>::value, "");
