//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter1, InputIterator Iter2>
//   requires HasEqualTo<Iter1::value_type, Iter2::value_type>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2);
//
// Introduced in C++14:
// template<InputIterator Iter1, InputIterator Iter2>
//   constexpr bool     // constexpr after c++17
//   equal(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2);

// We test the cartesian product, so we sometimes compare differently signed types
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-sign-compare
// ADDITIONAL_COMPILE_FLAGS(character-conversion-warnings): -Wno-character-conversion

// MSVC warning C4242: 'argument': conversion from 'int' to 'const _Ty', possible loss of data
// MSVC warning C4244: 'argument': conversion from 'wchar_t' to 'const _Ty', possible loss of data
// MSVC warning C4389: '==': signed/unsigned mismatch
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4242 /wd4244 /wd4389
// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

#include "sized_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class UnderlyingType, class Iter1>
struct Test {
  template <class Iter2>
  TEST_CONSTEXPR_CXX20 void operator()() {
    UnderlyingType a[]  = {0, 1, 2, 3, 4, 5};
    const unsigned s    = sizeof(a) / sizeof(a[0]);
    UnderlyingType b[s] = {0, 1, 2, 5, 4, 5};

    assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a)));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b)));

#if TEST_STD_VER >= 14
    assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a), std::equal_to<>()));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b), std::equal_to<>()));

    assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s)));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s - 1)));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b), Iter2(b + s)));

    assert(std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s), std::equal_to<>()));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(a), Iter2(a + s - 1), std::equal_to<>()));
    assert(!std::equal(Iter1(a), Iter1(a + s), Iter2(b), Iter2(b + s), std::equal_to<>()));
#endif
  }
};

struct TestNarrowingEqualTo {
  template <class UnderlyingType>
  TEST_CONSTEXPR_CXX20 void operator()() {
    TEST_DIAGNOSTIC_PUSH
    // MSVC warning C4310: cast truncates constant value
    TEST_MSVC_DIAGNOSTIC_IGNORED(4310)

    UnderlyingType a[] = {
        UnderlyingType(0x1000),
        UnderlyingType(0x1001),
        UnderlyingType(0x1002),
        UnderlyingType(0x1003),
        UnderlyingType(0x1004)};
    UnderlyingType b[] = {
        UnderlyingType(0x1600),
        UnderlyingType(0x1601),
        UnderlyingType(0x1602),
        UnderlyingType(0x1603),
        UnderlyingType(0x1604)};

    TEST_DIAGNOSTIC_POP

    assert(std::equal(a, a + 5, b, std::equal_to<char>()));
#if TEST_STD_VER >= 14
    assert(std::equal(a, a + 5, b, b + 5, std::equal_to<char>()));
#endif
  }
};

template <class UnderlyingType, class TypeList>
struct TestIter2 {
  template <class Iter1>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(TypeList(), Test<UnderlyingType, Iter1>());
  }
};

struct AddressCompare {
  int i = 0;
  TEST_CONSTEXPR_CXX20 AddressCompare(int) {}

  operator char() { return static_cast<char>(i); }

  friend TEST_CONSTEXPR_CXX20 bool operator==(const AddressCompare& lhs, const AddressCompare& rhs) {
    return &lhs == &rhs;
  }

  friend TEST_CONSTEXPR_CXX20 bool operator!=(const AddressCompare& lhs, const AddressCompare& rhs) {
    return &lhs != &rhs;
  }
};

#if TEST_STD_VER >= 20
class trivially_equality_comparable {
public:
  constexpr trivially_equality_comparable(int i) : i_(i) {}
  bool operator==(const trivially_equality_comparable&) const = default;

private:
  int i_;
};

#endif

template <std::size_t N>
TEST_CONSTEXPR_CXX20 void test_vector_bool() {
  std::vector<bool> in(N, false);
  for (std::size_t i = 0; i < N; i += 2)
    in[i] = true;

  { // Test equal() with aligned bytes
    std::vector<bool> out = in;
    assert(std::equal(in.begin(), in.end(), out.begin()));
#if TEST_STD_VER >= 14
    assert(std::equal(in.begin(), in.end(), out.begin(), out.end()));
#endif
  }

  { // Test equal() with unaligned bytes
    std::vector<bool> out(N + 8);
    std::copy(in.begin(), in.end(), out.begin() + 4);
    assert(std::equal(in.begin(), in.end(), out.begin() + 4));
#if TEST_STD_VER >= 14
    assert(std::equal(in.begin(), in.end(), out.begin() + 4, out.end() - 4));
#endif
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::cpp17_input_iterator_list<int*>(), TestIter2<int, types::cpp17_input_iterator_list<int*> >());
  types::for_each(
      types::cpp17_input_iterator_list<char*>(), TestIter2<char, types::cpp17_input_iterator_list<char*> >());
  types::for_each(types::cpp17_input_iterator_list<AddressCompare*>(),
                  TestIter2<AddressCompare, types::cpp17_input_iterator_list<AddressCompare*> >());

  types::for_each(types::integral_types(), TestNarrowingEqualTo());

#if TEST_STD_VER >= 20
  types::for_each(
      types::cpp17_input_iterator_list<trivially_equality_comparable*>{},
      TestIter2<trivially_equality_comparable, types::cpp17_input_iterator_list<trivially_equality_comparable*>>{});
#endif

  { // Test vector<bool>::iterator optimization
    test_vector_bool<8>();
    test_vector_bool<19>();
    test_vector_bool<32>();
    test_vector_bool<49>();
    test_vector_bool<64>();
    test_vector_bool<199>();
    test_vector_bool<256>();
  }

  // Make sure std::equal behaves properly with std::vector<bool> iterators with custom size types.
  // See issue: https://llvm.org/PR126369.
  {
    //// Tests for std::equal with aligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(6, true, Alloc(1));
      std::vector<bool, Alloc> expected(8, true, Alloc(1));
      assert(std::equal(in.begin() + 4, in.end(), expected.begin() + 4));
    }
    { // Test the last word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(12, true, Alloc(1));
      std::vector<bool, Alloc> expected(16, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin()));
    }
    { // Test middle words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(24, true, Alloc(1));
      std::vector<bool, Alloc> expected(29, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin()));
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(12, true, Alloc(1));
      std::vector<bool, Alloc> expected(16, true, Alloc(1));
      assert(std::equal(in.begin() + 4, in.end(), expected.begin() + 4));
    }
    { // Test the last word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(24, true, Alloc(1));
      std::vector<bool, Alloc> expected(32, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin()));
    }
    { // Test middle words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(48, true, Alloc(1));
      std::vector<bool, Alloc> expected(55, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin()));
    }

    //// Tests for std::equal with unaligned bits

    { // Test the first (partial) word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(6, true, Alloc(1));
      std::vector<bool, Alloc> expected(8, true, Alloc(1));
      assert(std::equal(in.begin() + 4, in.end(), expected.begin()));
    }
    { // Test the last word for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(4, true, Alloc(1));
      std::vector<bool, Alloc> expected(8, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin() + 3));
    }
    { // Test middle words for uint8_t
      using Alloc = sized_allocator<bool, std::uint8_t, std::int8_t>;
      std::vector<bool, Alloc> in(16, true, Alloc(1));
      std::vector<bool, Alloc> expected(24, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin() + 4));
    }

    { // Test the first (partial) word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(12, true, Alloc(1));
      std::vector<bool, Alloc> expected(16, true, Alloc(1));
      assert(std::equal(in.begin() + 4, in.end(), expected.begin()));
    }
    { // Test the last word for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(12, true, Alloc(1));
      std::vector<bool, Alloc> expected(16, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin() + 3));
    }
    { // Test the middle words for uint16_t
      using Alloc = sized_allocator<bool, std::uint16_t, std::int16_t>;
      std::vector<bool, Alloc> in(32, true, Alloc(1));
      std::vector<bool, Alloc> expected(64, true, Alloc(1));
      assert(std::equal(in.begin(), in.end(), expected.begin() + 4));
    }
  }

  return true;
}

struct Base {};
struct Derived : virtual Base {};

struct TestTypes {
  template <class T>
  struct Test {
    template <class U>
    void operator()() {
      T a[] = {1, 2, 3, 4, 5, 6};
      U b[] = {1, 2, 3, 4, 5, 6};
      assert(std::equal(a, a + 6, b));
    }
  };

  template <class T>
  void operator()() {
    types::for_each(types::integer_types(), Test<T>());
  }
};

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  types::for_each(types::integer_types(), TestTypes());
  types::for_each(types::as_pointers<types::cv_qualified_versions<int> >(),
                  TestIter2<int, types::as_pointers<types::cv_qualified_versions<int> > >());
  types::for_each(types::as_pointers<types::cv_qualified_versions<char> >(),
                  TestIter2<char, types::as_pointers<types::cv_qualified_versions<char> > >());

  {
    Derived d;
    Derived* a[] = {&d, nullptr};
    Base* b[]    = {&d, nullptr};

    assert(std::equal(a, a + 2, b));
#if TEST_STD_VER >= 14
    assert(std::equal(a, a + 2, b, b + 2));
#endif
  }

  return 0;
}
