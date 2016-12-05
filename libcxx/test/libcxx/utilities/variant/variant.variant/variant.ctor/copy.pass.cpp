// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// variant(variant const&);

#include <type_traits>
#include <variant>

#include "test_macros.h"

struct NTCopy {
  constexpr NTCopy(int v) : value(v) {}
  NTCopy(const NTCopy &that) : value(that.value) {}
  NTCopy(NTCopy &&) = delete;
  int value;
};

static_assert(!std::is_trivially_copy_constructible<NTCopy>::value, "");
static_assert(std::is_copy_constructible<NTCopy>::value, "");

struct TCopy {
  constexpr TCopy(int v) : value(v) {}
  TCopy(TCopy const &) = default;
  TCopy(TCopy &&) = delete;
  int value;
};

static_assert(std::is_trivially_copy_constructible<TCopy>::value, "");

struct TCopyNTMove {
  constexpr TCopyNTMove(int v) : value(v) {}
  TCopyNTMove(const TCopyNTMove&) = default;
  TCopyNTMove(TCopyNTMove&& that) : value(that.value) { that.value = -1; }
  int value;
};

static_assert(std::is_trivially_copy_constructible<TCopyNTMove>::value, "");

void test_copy_ctor_sfinae() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NTCopy>;
    static_assert(!std::is_trivially_copy_constructible<V>::value, "");
    static_assert(std::is_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TCopy>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TCopyNTMove>;
    static_assert(std::is_trivially_copy_constructible<V>::value, "");
  }
}

void test_copy_ctor_basic() {
  {
    constexpr std::variant<int> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0);
    constexpr std::variant<int> v2 = v;
    static_assert(v2.index() == 0);
    static_assert(std::get<0>(v2) == 42);
  }
  {
    constexpr std::variant<int, long> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1);
    constexpr std::variant<int, long> v2 = v;
    static_assert(v2.index() == 1);
    static_assert(std::get<1>(v2) == 42);
  }
  {
    constexpr std::variant<TCopy> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0);
    constexpr std::variant<TCopy> v2(v);
    static_assert(v2.index() == 0);
    static_assert(std::get<0>(v2).value == 42);
  }
  {
    constexpr std::variant<int, TCopy> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1);
    constexpr std::variant<int, TCopy> v2(v);
    static_assert(v2.index() == 1);
    static_assert(std::get<1>(v2).value == 42);
  }
  {
    constexpr std::variant<TCopyNTMove> v(std::in_place_index<0>, 42);
    static_assert(v.index() == 0);
    constexpr std::variant<TCopyNTMove> v2(v);
    static_assert(v2.index() == 0);
    static_assert(std::get<0>(v2).value == 42);
  }
  {
    constexpr std::variant<int, TCopyNTMove> v(std::in_place_index<1>, 42);
    static_assert(v.index() == 1);
    constexpr std::variant<int, TCopyNTMove> v2(v);
    static_assert(v2.index() == 1);
    static_assert(std::get<1>(v2).value == 42);
  }
}

int main() {
  test_copy_ctor_basic();
  test_copy_ctor_sfinae();
}
