//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asan
// UNSUPPORTED: c++03

#include <cassert>
#include <string>
#include <vector>
#include <array>
#include "test_macros.h"
#include "asan_testing.h"
#include "min_allocator.h"

// This tests exists to check if strings work well with vector, as those
// may be partialy annotated, we cannot simply call
// is_contiguous_container_asan_correct, as it assumes that
// object memory inside is not annotated, so we check everything in a more careful way.

template <typename D>
void verify_inside(D const& d) {
  for (size_t i = 0; i < d.size(); ++i) {
    assert(is_string_asan_correct(d[i]));
  }
}

template <typename S, size_t N>
S get_s(char c) {
  S s;
  for (size_t i = 0; i < N; ++i)
    s.push_back(c);

  return s;
}

template <class C, class S>
void test_string() {
  size_t const N = sizeof(S) < 256 ? (4096 / sizeof(S)) : 16;

  {
    C d1a(1), d1b(N), d1c(N + 1), d1d(32 * N);
    verify_inside(d1a);
    verify_inside(d1b);
    verify_inside(d1c);
    verify_inside(d1d);
  }
  {
    C d2;
    for (size_t i = 0; i < 16 * N; ++i) {
      d2.push_back(get_s<S, 1>(i % 10 + 'a'));
      verify_inside(d2);
      d2.push_back(get_s<S, 222>(i % 10 + 'b'));
      verify_inside(d2);

      d2.erase(d2.cbegin());
      verify_inside(d2);
    }
  }
  {
    C d3;
    for (size_t i = 0; i < 16 * N; ++i) {
      d3.push_back(get_s<S, 1>(i % 10 + 'a'));
      verify_inside(d3);
      d3.push_back(get_s<S, 222>(i % 10 + 'b'));
      verify_inside(d3);

      d3.pop_back();
      verify_inside(d3);
    }
  }
  {
    C d4;
    for (size_t i = 0; i < 16 * N; ++i) {
      // When there is no SSO, all elements inside should not be poisoned,
      // so we can verify vector poisoning.
      d4.push_back(get_s<S, 333>(i % 10 + 'a'));
      verify_inside(d4);
      assert(is_contiguous_container_asan_correct(d4));
      d4.push_back(get_s<S, 222>(i % 10 + 'b'));
      verify_inside(d4);
      assert(is_contiguous_container_asan_correct(d4));
    }
  }
  {
    C d5;
    for (size_t i = 0; i < 5 * N; ++i) {
      // In d4 we never had poisoned memory inside vector.
      // Here we start with SSO, so part of the inside of the container,
      // will be poisoned.
      d5.push_back(S());
      verify_inside(d5);
    }
    for (size_t i = 0; i < d5.size(); ++i) {
      // We change the size to have long string.
      // Memory owne by vector should not be poisoned by string.
      d5[i].resize(1000);
      verify_inside(d5);
    }

    assert(is_contiguous_container_asan_correct(d5));

    d5.erase(d5.begin() + 2);
    verify_inside(d5);

    d5.erase(d5.end() - 2);
    verify_inside(d5);

    assert(is_contiguous_container_asan_correct(d5));
  }
  {
    C d6a;
    assert(is_contiguous_container_asan_correct(d6a));

    C d6b(N + 2, get_s<S, 1000>('a'));
    d6b.push_back(get_s<S, 1001>('b'));
    while (!d6b.empty()) {
      d6b.pop_back();
      assert(is_contiguous_container_asan_correct(d6b));
    }

    C d6c(N + 2, get_s<S, 1002>('c'));
    while (!d6c.empty()) {
      d6c.pop_back();
      assert(is_contiguous_container_asan_correct(d6c));
    }
  }
  {
    C d7(9 * N + 2);

    d7.insert(d7.begin() + 1, S());
    verify_inside(d7);

    d7.insert(d7.end() - 3, S());
    verify_inside(d7);

    d7.insert(d7.begin() + 2 * N, get_s<S, 1>('a'));
    verify_inside(d7);

    d7.insert(d7.end() - 2 * N, get_s<S, 1>('b'));
    verify_inside(d7);

    d7.insert(d7.begin() + 2 * N, 3 * N, get_s<S, 1>('c'));
    verify_inside(d7);

    // It may not be short for big element types, but it will be checked correctly:
    d7.insert(d7.end() - 2 * N, 3 * N, get_s<S, 2>('d'));
    verify_inside(d7);

    d7.erase(d7.begin() + 2);
    verify_inside(d7);

    d7.erase(d7.end() - 2);
    verify_inside(d7);
  }
}

template <class S>
void test_container() {
  test_string<std::vector<S, std::allocator<S>>, S>();
  test_string<std::vector<S, min_allocator<S>>, S>();
  test_string<std::vector<S, safe_allocator<S>>, S>();
}

int main(int, char**) {
  // Those tests support only types based on std::basic_string.
  test_container<std::string>();
  test_container<std::wstring>();
#if TEST_STD_VER >= 11
  test_container<std::u16string>();
  test_container<std::u32string>();
#endif
#if TEST_STD_VER >= 20
  test_container<std::u8string>();
#endif

  return 0;
}
