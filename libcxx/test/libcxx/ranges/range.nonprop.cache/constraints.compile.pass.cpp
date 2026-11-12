//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
//  requires is_object_v<T>
// class non-propagating-cache;

#include <ranges>

template<template<class...> class T, class ...Args>
concept well_formed = requires {
  typename T<Args...>;
};

struct T { };
static_assert( well_formed<std::ranges::__non_propagating_cache, int>);
static_assert( well_formed<std::ranges::__non_propagating_cache, T>);
static_assert( well_formed<std::ranges::__non_propagating_cache, void (*)()>);
static_assert(!well_formed<std::ranges::__non_propagating_cache, void>);
static_assert(!well_formed<std::ranges::__non_propagating_cache, T&>);
static_assert(!well_formed<std::ranges::__non_propagating_cache, void()>);
