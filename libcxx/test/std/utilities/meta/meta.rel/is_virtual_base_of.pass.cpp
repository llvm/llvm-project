//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// These compilers don't support __builtin_is_virtual_base_of yet.
// UNSUPPORTED: clang-17, clang-18, clang-19, gcc-14, apple-clang-16, apple-clang-17

// <type_traits>

// std::is_virtual_base_of

#include <type_traits>
#include <cassert>

template <bool expected, class Base, class Derived>
void test() {
  // Test the type of the variables
  {
    static_assert(std::is_same_v<bool const, decltype(std::is_virtual_base_of<Base, Derived>::value)>);
    static_assert(std::is_same_v<bool const, decltype(std::is_virtual_base_of_v<Base, Derived>)>);
  }

  // Test their value
  {
    static_assert(std::is_virtual_base_of<Base, Derived>::value == expected);
    static_assert(std::is_virtual_base_of<const Base, Derived>::value == expected);
    static_assert(std::is_virtual_base_of<Base, const Derived>::value == expected);
    static_assert(std::is_virtual_base_of<const Base, const Derived>::value == expected);

    static_assert(std::is_virtual_base_of_v<Base, Derived> == expected);
    static_assert(std::is_virtual_base_of_v<const Base, Derived> == expected);
    static_assert(std::is_virtual_base_of_v<Base, const Derived> == expected);
    static_assert(std::is_virtual_base_of_v<const Base, const Derived> == expected);
  }

  // Check the relationship with is_base_of. If it's not a base of, it can't be a virtual base of.
  { static_assert(!std::is_base_of_v<Base, Derived> ? !std::is_virtual_base_of_v<Base, Derived> : true); }

  // Make sure they can be referenced at runtime
  {
    bool const& a = std::is_virtual_base_of<Base, Derived>::value;
    bool const& b = std::is_virtual_base_of_v<Base, Derived>;
    assert(a == expected);
    assert(b == expected);
  }
}

struct Incomplete;
struct Unrelated {};
union IncompleteUnion;
union Union {
  int i;
  float f;
};

class Base {};
class Derived : Base {};
class Derived2 : Base {};
class Derived2a : Derived {};
class Derived2b : Derived {};
class Derived3Virtual : virtual Derived2a, virtual Derived2b {};

struct DerivedTransitiveViaNonVirtual : Derived3Virtual {};
struct DerivedTransitiveViaVirtual : virtual Derived3Virtual {};

template <typename T>
struct CrazyDerived : T {};
template <typename T>
struct CrazyDerivedVirtual : virtual T {};

struct DerivedPrivate : private virtual Base {};
struct DerivedProtected : protected virtual Base {};
struct DerivedPrivatePrivate : private DerivedPrivate {};
struct DerivedPrivateProtected : private DerivedProtected {};
struct DerivedProtectedPrivate : protected DerivedProtected {};
struct DerivedProtectedProtected : protected DerivedProtected {};
struct DerivedTransitivePrivate : private Derived, private Derived2 {};

int main(int, char**) {
  // Test with non-virtual inheritance
  {
    test<false, Base, Base>();
    test<false, Base, Derived>();
    test<false, Base, Derived2>();
    test<false, Derived, DerivedTransitivePrivate>();
    test<false, Derived, Base>();
    test<false, Incomplete, Derived>();

    // Derived must be a complete type if Base and Derived are non-union class types
    // test<false, Base, Incomplete>();
  }

  // Test with virtual inheritance
  {
    test<false, Base, Derived3Virtual>();
    test<false, Derived, Derived3Virtual>();
    test<true, Derived2b, Derived3Virtual>();
    test<true, Derived2a, Derived3Virtual>();
    test<true, Base, DerivedPrivate>();
    test<true, Base, DerivedProtected>();
    test<true, Base, DerivedPrivatePrivate>();
    test<true, Base, DerivedPrivateProtected>();
    test<true, Base, DerivedProtectedPrivate>();
    test<true, Base, DerivedProtectedProtected>();
    test<true, Derived2a, DerivedTransitiveViaNonVirtual>();
    test<true, Derived2b, DerivedTransitiveViaNonVirtual>();
    test<true, Derived2a, DerivedTransitiveViaVirtual>();
    test<true, Derived2b, DerivedTransitiveViaVirtual>();
    test<false, Base, CrazyDerived<Base>>();
    test<false, CrazyDerived<Base>, Base>();
    test<true, Base, CrazyDerivedVirtual<Base>>();
    test<false, CrazyDerivedVirtual<Base>, Base>();
  }

  // Test unrelated types
  {
    test<false, Base&, Derived&>();
    test<false, Base[3], Derived[3]>();
    test<false, Unrelated, Derived>();
    test<false, Base, Unrelated>();
    test<false, Base, void>();
    test<false, void, Derived>();
  }

  // Test scalar types
  {
    test<false, int, Base>();
    test<false, int, Derived>();
    test<false, int, Incomplete>();
    test<false, int, int>();

    test<false, Base, int>();
    test<false, Derived, int>();
    test<false, Incomplete, int>();

    test<false, int[], int[]>();
    test<false, long, int>();
    test<false, int, long>();
  }

  // Test unions
  {
    test<false, Union, Union>();
    test<false, IncompleteUnion, IncompleteUnion>();
    test<false, Union, IncompleteUnion>();
    test<false, IncompleteUnion, Union>();
    test<false, Incomplete, IncompleteUnion>();
    test<false, IncompleteUnion, Incomplete>();
    test<false, Unrelated, IncompleteUnion>();
    test<false, IncompleteUnion, Unrelated>();
    test<false, int, IncompleteUnion>();
    test<false, IncompleteUnion, int>();
    test<false, Unrelated, Union>();
    test<false, Union, Unrelated>();
    test<false, int, Unrelated>();
    test<false, Union, int>();
  }

  return 0;
}
