//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// T& unique_ptr::operator[](size_t) const

#include <memory>
#include <cassert>
#include <type_traits>
#include <array>

#include "test_macros.h"
#include "type_algorithms.h"

static int next = 0;
struct EnumeratedDefaultCtor {
  EnumeratedDefaultCtor() : value(0) { value = ++next; }
  int value;
};

template <std::size_t Size>
struct WithTrivialDtor {
  std::array<char, Size> padding = {'x'};
  TEST_CONSTEXPR_CXX23 friend bool operator==(WithTrivialDtor const& x, WithTrivialDtor const& y) {
    return x.padding == y.padding;
  }
};

template <std::size_t Size>
struct WithNonTrivialDtor {
  std::array<char, Size> padding = {'x'};
  TEST_CONSTEXPR_CXX23 friend bool operator==(WithNonTrivialDtor const& x, WithNonTrivialDtor const& y) {
    return x.padding == y.padding;
  }
  TEST_CONSTEXPR_CXX23 ~WithNonTrivialDtor() {}
};

template <class T>
struct CustomDeleter : std::default_delete<T> {};

struct NoopDeleter {
  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

TEST_CONSTEXPR_CXX23 bool test() {
  // Basic test
  {
    std::unique_ptr<int[]> p(new int[3]);
    {
      int& result = p[0];
      result      = 0;
    }
    {
      int& result = p[1];
      result      = 1;
    }
    {
      int& result = p[2];
      result      = 2;
    }

    assert(p[0] == 0);
    assert(p[1] == 1);
    assert(p[2] == 2);
  }

  // Ensure that the order of access is correct after initializing a unique_ptr but
  // before actually modifying any of its elements. The implementation would have to
  // really try for this not to be the case, but we still check it.
  //
  // This requires assigning known values to the elements when they are first constructed,
  // which requires global state.
  {
    if (!TEST_IS_CONSTANT_EVALUATED) {
      std::unique_ptr<EnumeratedDefaultCtor[]> p(new EnumeratedDefaultCtor[3]);
      assert(p[0].value == 1);
      assert(p[1].value == 2);
      assert(p[2].value == 3);
    }
  }

  // Make sure operator[] is const-qualified
  {
    std::unique_ptr<int[]> const p(new int[3]);
    p[0] = 42;
    assert(p[0] == 42);
  }

  // Make sure we properly handle types with trivial and non-trivial destructors of different
  // sizes. This is relevant because some implementations may want to use properties of the
  // ABI like array cookies and these properties often depend on e.g. the triviality of T's
  // destructor, T's size and so on.
#if TEST_STD_VER >= 20 // this test is too painful to write before C++20
  {
    using TrickyCookieTypes = types::type_list<
        WithTrivialDtor<1>,
        WithTrivialDtor<2>,
        WithTrivialDtor<3>,
        WithTrivialDtor<4>,
        WithTrivialDtor<8>,
        WithTrivialDtor<16>,
        WithTrivialDtor<256>,
        WithNonTrivialDtor<1>,
        WithNonTrivialDtor<2>,
        WithNonTrivialDtor<3>,
        WithNonTrivialDtor<4>,
        WithNonTrivialDtor<8>,
        WithNonTrivialDtor<16>,
        WithNonTrivialDtor<256>>;
    types::for_each(TrickyCookieTypes(), []<class T> {
      // Array allocated with `new T[n]`, default deleter
      {
        std::unique_ptr<T[], std::default_delete<T[]>> p(new T[3]);
        assert(p[0] == T());
        assert(p[1] == T());
        assert(p[2] == T());
      }

      // Array allocated with `new T[n]`, custom deleter
      {
        std::unique_ptr<T[], CustomDeleter<T[]>> p(new T[3]);
        assert(p[0] == T());
        assert(p[1] == T());
        assert(p[2] == T());
      }

      // Array not allocated with `new T[n]`, custom deleter
      //
      // This test aims to ensure that the implementation doesn't try to use an array cookie
      // when there is none.
      {
        T array[50] = {};
        std::unique_ptr<T[], NoopDeleter> p(&array[0]);
        assert(p[0] == T());
        assert(p[1] == T());
        assert(p[2] == T());
      }
    });
  }
#endif // C++20

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}
