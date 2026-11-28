//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// PROTOTYPE: P3866R0 (V2: An Evolution Path for the Standard Library)
//
// Check how the standard-release namespaces map to std entities, pretending that std::vector2
// (a std::vector with a size-based layout) was introduced in C++29.

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <vector>

// The recommended usage pattern from the paper.
namespace cpp = std::cpp29;

// The revised component: in C++29, `vector` means `vector2`.
static_assert(std::is_same_v<cpp::vector<int>, std::vector2<int>>);
static_assert(std::is_same_v<cpp::vector<int, std::allocator<int>>, std::vector2<int>>);
static_assert(std::is_same_v<cpp::pmr::vector<int>, std::pmr::vector2<int>>);

// Older release namespaces pin the older version.
static_assert(std::is_same_v<std::cpp11::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp14::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp17::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp20::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp23::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp26::vector<int>, std::vector<int>>);
static_assert(std::is_same_v<std::cpp26::pmr::vector<int>, std::pmr::vector<int>>);

// vector<bool> is not special in vector2.
static_assert(std::is_same_v<cpp::vector<bool>::reference, bool&>);
static_assert(!std::is_same_v<std::cpp26::vector<bool>::reference, bool&>);

// Components that were never revised are found through the using-directive in each release namespace.
static_assert(std::is_same_v<cpp::string, std::string>);
static_assert(std::is_same_v<cpp::size_t, std::size_t>);
static_assert(std::is_same_v<cpp::allocator<int>, std::allocator<int>>);
static_assert(std::is_same_v<cpp::unique_ptr<int>, std::unique_ptr<int>>);

// Nested namespaces: std::cpp29::pmr has to exist to override `pmr::vector`, and it then has to re-expose
// everything else in std::pmr explicitly, otherwise it would hide it.
static_assert(std::is_same_v<cpp::pmr::polymorphic_allocator<int>, std::pmr::polymorphic_allocator<int>>);
static_assert(std::is_same_v<cpp::pmr::string, std::pmr::string>);

// Free functions are found through the using-directive too, and overload resolution picks the vector2 overloads.
void functions() {
  cpp::vector<int> v = {1, 2, 3, 2};
  cpp::vector<int> w;
  cpp::erase(v, 2);
  cpp::erase_if(v, [](int i) { return i == 3; });
  cpp::swap(v, w);
  (void)(v == w);
  (void)(v <=> w);

  // ADL works as usual since the associated namespace of std::vector2 is std.
  swap(v, w);
}

// Class template argument deduction goes through the alias template (C++20 CTAD for alias templates).
void ctad() {
  cpp::vector v = {1, 2, 3};
  static_assert(std::is_same_v<decltype(v), std::vector2<int>>);

  int arr[] = {1, 2, 3};
  cpp::vector w(arr, arr + 3);
  static_assert(std::is_same_v<decltype(w), std::vector2<int>>);

  std::cpp26::vector x = {1, 2, 3};
  static_assert(std::is_same_v<decltype(x), std::vector<int>>);
}

// Template argument deduction sees through the alias.
template <class T>
constexpr bool takes_cpp_vector(cpp::vector<T> const&) {
  return true;
}
static_assert(takes_cpp_vector(std::vector2<int>{}));
