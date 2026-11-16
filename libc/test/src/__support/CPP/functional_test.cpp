//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for cpp::function_ref.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/cstddef/nullptr_t.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/CPP/type_traits.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::function_ref;
using LIBC_NAMESPACE::cpp::is_assignable_v;
using LIBC_NAMESPACE::cpp::is_constructible_v;
using LIBC_NAMESPACE::cpp::nullptr_t;

static int forty_two() { return 42; }
static int seven() { return 7; }
static int add(int a, int b) { return a + b; }

static int (*get_forty_two())() { return &forty_two; }

namespace {

struct Functor {
  int operator()() const { return 0; }
};

struct Counter {
  int calls = 0;
  int operator()() { return ++calls; }
};

} // namespace

using IntFn = function_ref<int()>;
using IntIntFn = function_ref<int(int)>;

// Like std::function_ref, there is no empty state.
static_assert(!is_constructible_v<IntFn>);
static_assert(!is_constructible_v<IntFn, nullptr_t>);
static_assert(!is_constructible_v<bool, IntFn>);

// Callables with incompatible signatures are rejected.
static_assert(!is_constructible_v<IntFn, int (*)(int)>);
static_assert(!is_constructible_v<IntIntFn, int (*)()>);
static_assert(!is_constructible_v<function_ref<void *()>, int (*)()>);

// Assignment from anything other than another function_ref or a function
// pointer is deleted because it would dangle if the right-hand side is a
// temporary (e.g. `f = [&] {...};`). Like std::function_ref, lvalue callables
// are rejected too.
static_assert(is_assignable_v<IntFn &, const IntFn &>);
static_assert(is_assignable_v<IntFn &, int (*)()>);
static_assert(is_assignable_v<IntFn &, int (&)()>);
static_assert(is_constructible_v<IntFn, Functor>);
static_assert(!is_assignable_v<IntFn &, Functor>);
static_assert(!is_assignable_v<IntFn &, Functor &>);
static_assert(!is_assignable_v<IntFn &, nullptr_t>);

TEST(LlvmLibcFunctionRefTest, Lambda) {
  int offset = 10;
  auto lambda = [&offset](int x) { return x + offset; };
  IntIntFn f = lambda;
  EXPECT_EQ(f(1), 11);
  offset = 20;
  EXPECT_EQ(f(1), 21);
}

TEST(LlvmLibcFunctionRefTest, TemporaryLambdaArgument) {
  auto call = [](IntIntFn f) { return f(5); };
  EXPECT_EQ(call([](int x) { return x * 2; }), 10);
}

TEST(LlvmLibcFunctionRefTest, StatefulFunctor) {
  Counter counter;
  IntFn f = counter;
  EXPECT_EQ(f(), 1);
  function_ref<void()> g = counter; // The result is discarded.
  g();
  EXPECT_EQ(counter.calls, 2); // Refers to counter, not a copy.
}

TEST(LlvmLibcFunctionRefTest, Function) {
  IntFn f = forty_two;
  EXPECT_EQ(f(), 42);

  function_ref<int(int, int)> g = add;
  EXPECT_EQ(g(2, 3), 5);
}

TEST(LlvmLibcFunctionRefTest, FunctionPointer) {
  int (*ptr)() = &forty_two;
  IntFn f = ptr;
  // The pointer is captured by value, so changing the variable must not affect
  // the function_ref.
  ptr = &seven;
  EXPECT_EQ(f(), 42);
}

TEST(LlvmLibcFunctionRefTest, TemporaryFunctionPointer) {
  IntFn f = &forty_two;
  EXPECT_EQ(f(), 42);
  IntFn g = get_forty_two();
  EXPECT_EQ(g(), 42);
}

TEST(LlvmLibcFunctionRefTest, Copy) {
  IntFn f = forty_two;
  IntFn g = f;
  f = seven;
  EXPECT_EQ(g(), 42);
}

TEST(LlvmLibcFunctionRefTest, Assignment) {
  IntFn f = forty_two;
  IntFn g = seven;
  f = g;
  EXPECT_EQ(f(), 7);
  f = &forty_two;
  EXPECT_EQ(f(), 42);
}

TEST(LlvmLibcFunctionRefTest, ReturnConversion) {
  function_ref<long()> f = forty_two;
  EXPECT_EQ(f(), 42L);

  // Return values are discarded for void signatures.
  function_ref<void()> g = forty_two;
  g();
}

TEST(LlvmLibcFunctionRefTest, ReferenceParameter) {
  auto inc = [](int &x) { ++x; };
  function_ref<void(int &)> f = inc;
  int i = 1;
  f(i);
  EXPECT_EQ(i, 2);
}
