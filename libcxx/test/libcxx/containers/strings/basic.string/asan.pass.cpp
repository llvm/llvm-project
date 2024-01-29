//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asan
// UNSUPPORTED: c++03

// Basic test if ASan annotations work for basic_string.

#include <string>
#include <cassert>
#include <cstdlib>

#include "asan_testing.h"
#include "min_allocator.h"
#include "test_iterators.h"
#include "test_macros.h"

extern "C" void __sanitizer_set_death_callback(void (*callback)(void));

void do_exit() { exit(0); }

int main(int, char**) {
  {
    typedef cpp17_input_iterator<char*> MyInputIter;
    // Should not trigger ASan.
    std::basic_string<char, std::char_traits<char>, safe_allocator<char>> v;
    char i[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'a', 'b', 'c', 'd', 'e',
                'f', 'g', 'h', 'i', 'j', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};

    v.insert(v.begin(), MyInputIter(i), MyInputIter(i + 29));
    assert(v[0] == 'a');
    assert(is_string_asan_correct(v));
  }

  __sanitizer_set_death_callback(do_exit);
  {
    using T     = char;
    using C     = std::basic_string<T, std::char_traits<T>, safe_allocator<T>>;
    const T t[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'a', 'b', 'c', 'd', 'e',
                   'f', 'g', 'h', 'i', 'j', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'};
    C c(std::begin(t), std::end(t));
    assert(is_string_asan_correct(c));
    assert(__sanitizer_verify_contiguous_container(c.data(), c.data() + c.size() + 1, c.data() + c.capacity() + 1) !=
           0);
    T foo = c[c.size() + 1]; // should trigger ASAN and call do_exit().
    assert(false);           // if we got here, ASAN didn't trigger
    ((void)foo);

    return 0;
  }
}
