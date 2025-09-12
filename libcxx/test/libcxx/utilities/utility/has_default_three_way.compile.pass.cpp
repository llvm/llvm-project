//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__utility/default_three_way_comparator.h>
#include <string>
#include <vector>

static_assert(std::__has_default_three_way_comparator<int, int>::value);
static_assert(std::__has_default_three_way_comparator<int, long>::value);
static_assert(std::__has_default_three_way_comparator<long, int>::value);
static_assert(std::__has_default_three_way_comparator<long, long>::value);
static_assert(std::__has_default_three_way_comparator<std::string, std::string>::value);

#if __has_builtin(__builtin_lt_synthesises_from_spaceship)
static_assert(std::__has_default_three_way_comparator<const std::string&, const std::string&>::value);
static_assert(std::__has_default_three_way_comparator<const std::string&, const std::string_view&>::value);
static_assert(std::__has_default_three_way_comparator<std::string, std::string_view>::value);
static_assert(std::__has_default_three_way_comparator<const std::string&, const char*>::value);
static_assert(std::__has_default_three_way_comparator<std::string, const char*>::value);
static_assert(!std::__has_default_three_way_comparator<const std::string&, const wchar_t*>::value);

static_assert(std::__has_default_three_way_comparator<const std::vector<int>&, const std::vector<int>&>::value);

struct MyStruct {
  int i;

  friend auto operator<=>(MyStruct, MyStruct) = default;
};

static_assert(std::__has_default_three_way_comparator<const MyStruct&, const MyStruct&>::value);
#endif
