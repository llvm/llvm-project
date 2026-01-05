//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <functional>

// Type of `std::not_fn<NTTP>()` is always empty.

#include <functional>
#include <type_traits>

struct NonEmptyFunctionObject {
  bool val = true;
  bool operator()() const;
};

bool func();

struct SomeClass {
  bool member_object;
  bool member_function();
};

using ResultWithEmptyFuncObject = decltype(std::not_fn<std::false_type{}>());
static_assert(std::is_empty_v<ResultWithEmptyFuncObject>);

using ResultWithNotEmptyFuncObject = decltype(std::not_fn<NonEmptyFunctionObject{}>());
static_assert(std::is_empty_v<ResultWithNotEmptyFuncObject>);

using ResultWithFunctionPointer = decltype(std::not_fn<&func>());
static_assert(std::is_empty_v<ResultWithFunctionPointer>);

using ResultWithMemberObjectPointer = decltype(std::not_fn<&SomeClass::member_object>());
static_assert(std::is_empty_v<ResultWithMemberObjectPointer>);

using ResultWithMemberFunctionPointer = decltype(std::not_fn<&SomeClass::member_function>());
static_assert(std::is_empty_v<ResultWithMemberFunctionPointer>);
