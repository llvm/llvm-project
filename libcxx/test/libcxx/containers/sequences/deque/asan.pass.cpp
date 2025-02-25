//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asan

// <deque>

// reference operator[](size_type n);

#include "asan_testing.h"
#include <deque>
#include <cassert>
#include <cstdlib>

#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

extern "C" void __sanitizer_set_death_callback(void (*callback)(void));

void do_exit() { exit(0); }

int main(int, char**) {
  {
    typedef cpp17_input_iterator<int*> MyInputIter;
    // Should not trigger ASan.
    std::deque<int> v;
    int i[] = {42};
    v.insert(v.begin(), MyInputIter(i), MyInputIter(i + 1));
    assert(v[0] == 42);
    assert(is_double_ended_contiguous_container_asan_correct(v));
  }
  {
    typedef int T;
    typedef std::deque<T, min_allocator<T> > C;
    const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    assert(is_double_ended_contiguous_container_asan_correct(c));
  }
  {
    typedef char T;
    typedef std::deque<T, safe_allocator<T> > C;
    const T t[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    C c(std::begin(t), std::end(t));
    c.pop_front();
    assert(is_double_ended_contiguous_container_asan_correct(c));
  }
  __sanitizer_set_death_callback(do_exit);
  {
    typedef int T;
    typedef std::deque<T> C;
    const T t[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    C c(std::begin(t), std::end(t));
    assert(is_double_ended_contiguous_container_asan_correct(c));
    T* ptr = &c[0];
    for (size_t i = 0; i < (8 + sizeof(T) - 1) / sizeof(T); ++i)
      c.pop_front();
    *ptr           = 1;
    volatile T foo = c[c.size()]; // should trigger ASAN. Use volatile to prevent being optimized away.
    assert(false);                // if we got here, ASAN didn't trigger
    ((void)foo);
  }
}
