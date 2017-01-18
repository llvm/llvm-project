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

// variant(variant&&) noexcept(see below);

#include <type_traits>
#include <variant>

#include "test_macros.h"

struct NTMove {
  constexpr NTMove(int v) : value(v) {}
  NTMove(const NTMove &) = delete;
  NTMove(NTMove &&that) : value(that.value) { that.value = -1; }
  int value;
};

static_assert(!std::is_trivially_move_constructible<NTMove>::value, "");
static_assert(std::is_move_constructible<NTMove>::value, "");

struct TMove {
  constexpr TMove(int v) : value(v) {}
  TMove(const TMove &) = delete;
  TMove(TMove &&) = default;
  int value;
};

static_assert(std::is_trivially_move_constructible<TMove>::value, "");

struct TMoveNTCopy {
  constexpr TMoveNTCopy(int v) : value(v) {}
  TMoveNTCopy(const TMoveNTCopy& that) : value(that.value) {}
  TMoveNTCopy(TMoveNTCopy&&) = default;
  int value;
};

static_assert(std::is_trivially_move_constructible<TMoveNTCopy>::value, "");

void test_move_ctor_sfinae() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, NTMove>;
    static_assert(!std::is_trivially_move_constructible<V>::value, "");
    static_assert(std::is_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TMove>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = std::variant<int, TMoveNTCopy>;
    static_assert(std::is_trivially_move_constructible<V>::value, "");
  }
}

template <typename T>
struct Result { size_t index; T value; };

void test_move_ctor_basic() {
  {
    struct {
      constexpr Result<int> operator()() const {
        std::variant<int> v(std::in_place_index<0>, 42);
        std::variant<int> v2 = std::move(v);
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value == 42, "");
  }
  {
    struct {
      constexpr Result<long> operator()() const {
        std::variant<int, long> v(std::in_place_index<1>, 42);
        std::variant<int, long> v2 = std::move(v);
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMove> operator()() const {
        std::variant<TMove> v(std::in_place_index<0>, 42);
        std::variant<TMove> v2(std::move(v));
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMove> operator()() const {
        std::variant<int, TMove> v(std::in_place_index<1>, 42);
        std::variant<int, TMove> v2(std::move(v));
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMoveNTCopy> operator()() const {
        std::variant<TMoveNTCopy> v(std::in_place_index<0>, 42);
        std::variant<TMoveNTCopy> v2(std::move(v));
        return {v2.index(), std::get<0>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct {
      constexpr Result<TMoveNTCopy> operator()() const {
        std::variant<int, TMoveNTCopy> v(std::in_place_index<1>, 42);
        std::variant<int, TMoveNTCopy> v2(std::move(v));
        return {v2.index(), std::get<1>(std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
}

int main() {
  test_move_ctor_basic();
  test_move_ctor_sfinae();
}
