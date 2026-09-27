//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <functional>

// Type of `std::bind_front<NTTP>(/* no bound args */)` is empty.

#include <functional>
#include <type_traits>

struct NonEmptyFunctionObject {
  int val = true;
  void operator()() const;
};

void func();

struct SomeClass {
  long member_object;
  void member_function();
};

using ResultWithEmptyFuncObject = decltype(std::bind_front<std::integral_constant<int, 0>{}>());
static_assert(std::is_empty_v<ResultWithEmptyFuncObject>);

using ResultWithNotEmptyFuncObject = decltype(std::bind_front<NonEmptyFunctionObject{}>());
static_assert(std::is_empty_v<ResultWithNotEmptyFuncObject>);

using ResultWithFunctionPointer = decltype(std::bind_front<func>());
static_assert(std::is_empty_v<ResultWithFunctionPointer>);

using ResultWithMemberObjectPointer = decltype(std::bind_front<&SomeClass::member_object>());
static_assert(std::is_empty_v<ResultWithMemberObjectPointer>);

using ResultWithMemberFunctionPointer = decltype(std::bind_front<&SomeClass::member_function>());
static_assert(std::is_empty_v<ResultWithMemberFunctionPointer>);
