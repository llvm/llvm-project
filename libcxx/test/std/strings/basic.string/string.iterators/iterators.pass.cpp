//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <string>

// iterator       begin(); // constexpr since C++20
// iterator       end(); // constexpr since C++20
// const_iterator begin()  const; // constexpr since C++20
// const_iterator end()    const; // constexpr since C++20
// const_iterator cbegin() const; // constexpr since C++20
// const_iterator cend()   const; // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"

template <class C>
TEST_CONSTEXPR_CXX20 void test() {
  { // N3644 testing
    typename C::iterator ii1{}, ii2{};
    typename C::iterator ii4 = ii1;
    typename C::const_iterator cii{};

    assert(ii1 == ii2);
    assert(ii1 == ii4);

    assert(!(ii1 != ii2));

    assert((ii1 == cii));
    assert((cii == ii1));
    assert(!(ii1 != cii));
    assert(!(cii != ii1));
    assert(!(ii1 < cii));
    assert(!(cii < ii1));
    assert((ii1 <= cii));
    assert((cii <= ii1));
    assert(!(ii1 > cii));
    assert(!(cii > ii1));
    assert((ii1 >= cii));
    assert((cii >= ii1));
    assert(cii - ii1 == 0);
    assert(ii1 - cii == 0);
  }
  {
    C a;
    typename C::iterator i1 = a.begin();
    typename C::iterator i2;
    i2 = i1;
    assert(i1 == i2);
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test<std::string>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<std::wstring>();
#endif

#ifndef TEST_HAS_NO_CHAR8_T
  test<std::u8string>();
#endif

  test<std::u16string>();
  test<std::u32string>();

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif
  return 0;
}
