//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class seed_seq;

// template<class InputIterator>
//   seed_seq(InputIterator begin, InputIterator end);
// Mandates: iterator_traits<InputIterator>::value_type is an integer type.

#include <random>

void test()
{
  {
    bool a[2] = {true, false};
    std::seed_seq s(a, a+2); // OK
  }
  {
    double a[2] = {1, 2};
    std::seed_seq s(a, a+2); // expected-error@*:* {{Mandates: iterator_traits<InputIterator>::value_type is an integer type}}
        // expected-error@*:* {{invalid operands to binary expression ('double' and 'unsigned int')}}
  }
}
