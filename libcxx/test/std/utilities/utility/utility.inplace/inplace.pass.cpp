//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <utility>

// struct in_place_tag { in_place_tag() = delete; };
//
// using in_place_t = in_place_tag(&)(unspecified);
// template <class T>
//     using in_place_type_t = in_place_tag(&)(unspecified<T>);
// template <size_t N>
//     using in_place_index_t = in_place_tag(&)(unspecified<N>);
//
// in_place_tag in_place(unspecified);
//
// template <class T>;
// in_place_tag in_place(unspecified<T>);
//
// template <size_t N>
// in_place_tag in_place(unspecified<N>);

#include <utility>
#include <cassert>
#include <memory>

#include "test_macros.h"
#include "type_id.h"

template <class Tp>
struct CheckRet : std::false_type {};
template <class Arg>
struct CheckRet<std::in_place_tag(Arg)> : std::true_type {};

TypeID const* test_fn(std::in_place_t) { return &makeTypeID<std::in_place_t>(); }
template <class T>
TypeID const* test_fn(std::in_place_type_t<T>)
{ return &makeTypeID<std::in_place_type_t<T>>(); }

template <size_t I>
TypeID const* test_fn(std::in_place_index_t<I>)
{ return &makeTypeID<std::in_place_index_t<I>>(); }

// Concrete test overloads that don't have to be deduced.
template <class Tag>
TypeID const* concrete_test_fn(Tag) {  return &makeTypeID<Tag>(); }

template <class Tp>
bool check_tag_basic() {
  using RawTp = typename std::remove_reference<Tp>::type;
  static_assert(std::is_lvalue_reference<Tp>::value, "");
  static_assert(std::is_function<RawTp>::value, "");
  static_assert(CheckRet<RawTp>::value, "");
  auto concrete_fn = concrete_test_fn<Tp>;
  return test_fn((Tp)std::in_place) == &makeTypeID<Tp>()
      && concrete_fn(std::in_place) == &makeTypeID<Tp>();
}

int main() {
    // test in_place_tag
    {
        static_assert(!std::is_default_constructible<std::in_place_tag>::value, "");
    }
    // test in_place_t
    {
        using T = std::in_place_t;
        assert(check_tag_basic<std::in_place_t>());
        assert(test_fn((T)std::in_place) == &makeTypeID<T>());
    }
    // test in_place_type_t
    {
        using T1 = std::in_place_type_t<void>;
        using T2 = std::in_place_type_t<int>;
        using T3 = std::in_place_type_t<const int>;
        assert(check_tag_basic<T1>());
        assert(check_tag_basic<T2>());
        assert(check_tag_basic<T3>());
        static_assert(!std::is_same<T1, T2>::value && !std::is_same<T1, T3>::value, "");
        static_assert(!std::is_same<T2, T3>::value, "");
    }
    // test in_place_index_t
    {
        using T1 = std::in_place_index_t<0>;
        using T2 = std::in_place_index_t<1>;
        using T3 = std::in_place_index_t<static_cast<size_t>(-1)>;
        assert(check_tag_basic<T1>());
        assert(check_tag_basic<T2>());
        assert(check_tag_basic<T3>());
        static_assert(!std::is_same<T1, T2>::value && !std::is_same<T1, T3>::value, "");
        static_assert(!std::is_same<T2, T3>::value, "");
    }
}