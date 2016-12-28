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

// Clang 3.8 doesn't generate constexpr special members correctly.
// XFAIL: clang-3.8, apple-clang-7, apple-clang-8

// <variant>

// template <class ...Types> class variant;

// variant& operator=(variant const&);

#include <type_traits>
#include <variant>

#include "test_macros.h"

struct NTCopyAssign {
  constexpr NTCopyAssign(int v) : value(v) {}
  NTCopyAssign(const NTCopyAssign &) = default;
  NTCopyAssign(NTCopyAssign &&) = default;
  NTCopyAssign &operator=(const NTCopyAssign &that) {
    value = that.value;
    return *this;
  };
  NTCopyAssign &operator=(NTCopyAssign &&) = delete;
  int value;
};

static_assert(!std::is_trivially_copy_assignable<NTCopyAssign>::value, "");
static_assert(std::is_copy_assignable<NTCopyAssign>::value, "");

struct TCopyAssign {
  constexpr TCopyAssign(int v) : value(v) {}
  TCopyAssign(const TCopyAssign &) = default;
  TCopyAssign(TCopyAssign &&) = default;
  TCopyAssign &operator=(const TCopyAssign &) = default;
  TCopyAssign &operator=(TCopyAssign &&) = delete;
  int value;
};

static_assert(std::is_trivially_copy_assignable<TCopyAssign>::value, "");

struct TCopyAssignNTMoveAssign {
  constexpr TCopyAssignNTMoveAssign(int v) : value(v) {}
  TCopyAssignNTMoveAssign(const TCopyAssignNTMoveAssign &) = default;
  TCopyAssignNTMoveAssign(TCopyAssignNTMoveAssign &&) = default;
  TCopyAssignNTMoveAssign &operator=(const TCopyAssignNTMoveAssign &) = default;
  TCopyAssignNTMoveAssign &operator=(TCopyAssignNTMoveAssign &&that) {
    value = that.value;
    that.value = -1;
    return *this;
  }
  int value;
};

static_assert(std::is_trivially_copy_assignable_v<TCopyAssignNTMoveAssign>);

void test_copy_assignment_sfinae() {
  {
    using V = std::variant<int, long>;
    static_assert(std::is_trivially_copy_assignable<V>::value, "");
  }
  {
    using V = std::variant<int, NTCopyAssign>;
    static_assert(!std::is_trivially_copy_assignable<V>::value, "");
    static_assert(std::is_copy_assignable<V>::value, "");
  }
  {
    using V = std::variant<int, TCopyAssign>;
    static_assert(std::is_trivially_copy_assignable<V>::value, "");
  }
  {
    using V = std::variant<int, TCopyAssignNTMoveAssign>;
    static_assert(std::is_trivially_copy_assignable<V>::value, "");
  }
}

template <typename T> struct Result { size_t index; T value; };

void test_copy_assignment_same_index() {
  {
    struct {
      constexpr Result<int> operator()() const {
        using V = std::variant<int>;
        V v(43);
        V v2(42);
        v = v2;
        return {v.index(), std::get<0>(v)};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0);
    static_assert(result.value == 42);
  }
  {
    struct {
      constexpr Result<long> operator()() const {
        using V = std::variant<int, long, unsigned>;
        V v(43l);
        V v2(42l);
        v = v2;
        return {v.index(), std::get<1>(v)};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1);
    static_assert(result.value == 42l);
  }
  {
    struct {
      constexpr Result<int> operator()() const {
        using V = std::variant<int, TCopyAssign, unsigned>;
        V v(std::in_place_type<TCopyAssign>, 43);
        V v2(std::in_place_type<TCopyAssign>, 42);
        v = v2;
        return {v.index(), std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1);
    static_assert(result.value == 42);
  }
  {
    struct {
      constexpr Result<int> operator()() const {
        using V = std::variant<int, TCopyAssignNTMoveAssign, unsigned>;
        V v(std::in_place_type<TCopyAssignNTMoveAssign>, 43);
        V v2(std::in_place_type<TCopyAssignNTMoveAssign>, 42);
        v = v2;
        return {v.index(), std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1);
    static_assert(result.value == 42);
  }
}

void test_copy_assignment_different_index() {
  {
    struct {
      constexpr Result<long> operator()() const {
        using V = std::variant<int, long, unsigned>;
        V v(43);
        V v2(42l);
        v = v2;
        return {v.index(), std::get<1>(v)};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1);
    static_assert(result.value == 42l);
  }
  {
    struct {
      constexpr Result<int> operator()() const {
        using V = std::variant<int, TCopyAssign, unsigned>;
        V v(std::in_place_type<unsigned>, 43);
        V v2(std::in_place_type<TCopyAssign>, 42);
        v = v2;
        return {v.index(), std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1);
    static_assert(result.value == 42);
  }
}

template <size_t NewIdx, class ValueType>
constexpr bool test_constexpr_assign_extension_imp(
    std::variant<long, void*, int>&& v, ValueType&& new_value)
{
  const std::variant<long, void*, int> cp(
      std::forward<ValueType>(new_value));
  v = cp;
  return v.index() == NewIdx &&
        std::get<NewIdx>(v) == std::get<NewIdx>(cp);
}

void test_constexpr_copy_assignment_extension() {
#ifdef _LIBCPP_VERSION
  using V = std::variant<long, void*, int>;
  static_assert(std::is_trivially_copyable<V>::value, "");
  static_assert(std::is_trivially_copy_assignable<V>::value, "");
  static_assert(test_constexpr_assign_extension_imp<0>(V(42l), 101l), "");
  static_assert(test_constexpr_assign_extension_imp<0>(V(nullptr), 101l), "");
  static_assert(test_constexpr_assign_extension_imp<1>(V(42l), nullptr), "");
  static_assert(test_constexpr_assign_extension_imp<2>(V(42l), 101), "");
#endif
}

int main() {
  test_copy_assignment_same_index();
  test_copy_assignment_different_index();
  test_copy_assignment_sfinae();
  test_constexpr_copy_assignment_extension();
}
