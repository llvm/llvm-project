//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <experimental/stacktrace>

#include <cassert>

/*
  (19.6.4.4) Comparisons [stacktrace.basic.cmp]

  template<class Allocator2>
  friend bool operator==(const basic_stacktrace& x,
                          const basic_stacktrace<Allocator2>& y) noexcept;

  template<class Allocator2>
  friend strong_ordering operator<=>(const basic_stacktrace& x,
                                      const basic_stacktrace<Allocator2>& y) noexcept;
*/

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test1() { return std::stacktrace::current(); }

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2a() { return test1(); }

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::stacktrace test2b() { return test1(); }

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  /*    
    template<class Allocator2>
    friend bool operator==(const basic_stacktrace& x, const basic_stacktrace<Allocator2>& y) noexcept;
    Returns: equal(x.begin(), x.end(), y.begin(), y.end()).
  */
  auto st1a = test1(); // [test1, main, ...]
  assert(st1a == st1a);

  auto st1b = st1a;
  assert(st1a == st1b);

  auto st2a = test2a(); // [test1, test2a, main, ...]
  assert(st1a != st2a);

  std::stacktrace empty; // []
  assert(st1a != empty);
  assert(st2a != empty);

  assert(st2a.size() > st1a.size());
  assert(st1a.size() > empty.size());

  auto st2b = test2b(); // [test1, test2b, main, ...]
  assert(st2a.size() == st2b.size());
  assert(st2a != st2b);

  /*    
    template<class Allocator2>
    friend strong_ordering
    operator<=>(const basic_stacktrace& x, const basic_stacktrace<Allocator2>& y) noexcept;

    Returns: x.size() <=> y.size() if x.size() != y.size();
    lexicographical_compare_three_way(x.begin(), x.end(), y.begin(), y.end()) otherwise.
  */

  // empty:  []
  // st1a:   [test1, main, ...]
  // st1b:   [test1, main, ...] (copy of st1a)
  // st2a:   [test1, test2a, main:X, ...]
  // st2b:   [test1, test2b, main:Y, ...], Y > X

  assert(std::strong_ordering::equal == empty <=> empty);
  assert(std::strong_ordering::less == empty <=> st1a);
  assert(std::strong_ordering::less == empty <=> st1b);
  assert(std::strong_ordering::less == empty <=> st2a);
  assert(std::strong_ordering::less == empty <=> st2b);

  assert(std::strong_ordering::greater == st1a <=> empty);
  assert(std::strong_ordering::equal == st1a <=> st1a);
  assert(std::strong_ordering::equal == st1a <=> st1b);
  assert(std::strong_ordering::less == st1a <=> st2a);
  assert(std::strong_ordering::less == st1a <=> st2b);

  assert(std::strong_ordering::greater == st1b <=> empty);
  assert(std::strong_ordering::equal == st1b <=> st1a);
  assert(std::strong_ordering::equal == st1b <=> st1b);
  assert(std::strong_ordering::less == st1b <=> st2a);
  assert(std::strong_ordering::less == st1b <=> st2b);

  assert(std::strong_ordering::greater == st2a <=> empty);
  assert(std::strong_ordering::greater == st2a <=> st1a);
  assert(std::strong_ordering::greater == st2a <=> st1b);
  assert(std::strong_ordering::equal == st2a <=> st2a);
  assert(std::strong_ordering::less == st2a <=> st2b);

  assert(std::strong_ordering::greater == st2b <=> empty);
  assert(std::strong_ordering::greater == st2b <=> st1a);
  assert(std::strong_ordering::greater == st2b <=> st1b);
  assert(std::strong_ordering::greater == st2b <=> st2a);
  assert(std::strong_ordering::equal == st2b <=> st2b);

  return 0;
}
