//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// static constexpr decltype(auto) value = (X);
// using type = constant_wrapper;
// using value_type = decltype(X);

#include <concepts>
#include <utility>

static_assert(std::constant_wrapper<42>::value == 42);
static_assert(std::same_as<decltype(std::constant_wrapper<42>::value), const int>);
static_assert(std::same_as<std::constant_wrapper<42>::type, std::constant_wrapper<42>>);
static_assert(std::same_as<std::constant_wrapper<42>::value_type, int>);

struct S {
  int member = 42;
};

static_assert(std::constant_wrapper<S{5}>::value.member == 5);
static_assert(std::same_as<decltype(std::constant_wrapper<S{5}>::value), const S&>);
static_assert(std::same_as<std::constant_wrapper<S{5}>::type, std::constant_wrapper<S{5}>>);
static_assert(std::same_as<std::constant_wrapper<S{5}>::value_type, S>);

template <auto V>
consteval bool value_ref_to_template_parameter_object() {
  return &V == &std::constant_wrapper<V>::value;
}

static_assert(value_ref_to_template_parameter_object<S{5}>());

constexpr int arr[] = {1, 2, 3, 4, 5};

static_assert(std::constant_wrapper<arr>::value == arr);
static_assert(std::same_as<std::constant_wrapper<arr>::type, std::constant_wrapper<arr>>);
static_assert(std::same_as<std::constant_wrapper<arr>::value_type, const int*>);
