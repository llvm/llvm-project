//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept uniform_random_bit_generator = // see below

#include <random>

#include "test_macros.h"

static_assert(std::uniform_random_bit_generator<
              std::linear_congruential_engine<unsigned int, 0U, 1U, 2U> >);

#ifndef TEST_HAS_NO_INT128
static_assert(std::uniform_random_bit_generator<
              std::subtract_with_carry_engine<__uint128_t, 1U, 2U, 3U> >);
#endif

// Not invocable
static_assert(!std::uniform_random_bit_generator<void>);
static_assert(!std::uniform_random_bit_generator<int>);
static_assert(!std::uniform_random_bit_generator<int[10]>);
static_assert(!std::uniform_random_bit_generator<int*>);
static_assert(!std::uniform_random_bit_generator<const int*>);
static_assert(!std::uniform_random_bit_generator<volatile int*>);
static_assert(!std::uniform_random_bit_generator<const volatile int*>);
static_assert(!std::uniform_random_bit_generator<int&>);
static_assert(!std::uniform_random_bit_generator<const int&>);
static_assert(!std::uniform_random_bit_generator<volatile int&>);
static_assert(!std::uniform_random_bit_generator<const volatile int&>);
static_assert(!std::uniform_random_bit_generator<int&&>);
static_assert(!std::uniform_random_bit_generator<const int&&>);
static_assert(!std::uniform_random_bit_generator<volatile int&&>);
static_assert(!std::uniform_random_bit_generator<const volatile int&&>);

struct Empty {};
static_assert(!std::uniform_random_bit_generator<Empty>);

namespace WrongReturnType {
using FunctionPointer = void (*)();
static_assert(!std::uniform_random_bit_generator<FunctionPointer>);

using FunctionReference = int (&)();
static_assert(!std::uniform_random_bit_generator<FunctionReference>);

struct FunctionObject {
  unsigned long* operator()();
};
static_assert(!std::uniform_random_bit_generator<FunctionObject>);
static_assert(!std::uniform_random_bit_generator<unsigned int Empty::*>);
static_assert(!std::uniform_random_bit_generator<unsigned short (Empty::*)()>);
} // namespace WrongReturnType

namespace NoMinOrMax {
using FunctionPointer = unsigned int (*)();
static_assert(!std::uniform_random_bit_generator<FunctionPointer>);

using FunctionReference = unsigned long long (&)();
static_assert(!std::uniform_random_bit_generator<FunctionReference>);

struct FunctionObject {
  unsigned char operator()();
};
static_assert(!std::uniform_random_bit_generator<FunctionObject>);
} // namespace NoMinOrMax

namespace OnlyMinIsRight {
struct NoMax {
  unsigned char operator()();

  static unsigned char min();
};
static_assert(!std::uniform_random_bit_generator<NoMax>);

struct MaxHasWrongReturnType {
  unsigned char operator()();

  static unsigned char min();
  static unsigned int max();
};

static_assert(!std::uniform_random_bit_generator<MaxHasWrongReturnType>);
} // namespace OnlyMinIsRight

namespace OnlyMaxIsRight {
struct NoMin {
  unsigned char operator()();

  static unsigned char max();
};
static_assert(!std::uniform_random_bit_generator<NoMin>);

struct MinHasWrongReturnType {
  unsigned char operator()();

  static unsigned int min();
  static unsigned char max();
};

static_assert(!std::uniform_random_bit_generator<MinHasWrongReturnType>);
} // namespace OnlyMaxIsRight

namespace MinNotLessMax {
struct NotConstexpr {
  unsigned char operator()();

  static unsigned char min();
  static unsigned char max();
};
static_assert(!std::uniform_random_bit_generator<NotConstexpr>);

struct MinEqualsMax {
  unsigned char operator()();

  static constexpr unsigned char min() { return 0; }
  static constexpr unsigned char max() { return 0; }
};
static_assert(!std::uniform_random_bit_generator<MinEqualsMax>);

struct MaxLessThanMin {
  unsigned char operator()();

  static constexpr unsigned char min() { return 1; }
  static constexpr unsigned char max() { return 0; }
};
static_assert(!std::uniform_random_bit_generator<MaxLessThanMin>);
} // namespace MinNotLessMax

struct Works {
  unsigned char operator()();

  static constexpr unsigned char min() { return 0; }
  static constexpr unsigned char max() { return 1; }
};
static_assert(std::uniform_random_bit_generator<Works>);

int main(int, char**) { return 0; }
