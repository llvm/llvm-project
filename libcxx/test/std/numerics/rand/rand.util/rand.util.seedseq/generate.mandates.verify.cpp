//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class seed_seq;

// template<class RandomAccessIterator>
//     void generate(RandomAccessIterator begin, RandomAccessIterator end);

// Check the following requirement: https://eel.is/c++draft/rand.util.seedseq#7
//
//  Mandates: iterator_traits<RandomAccessIterator>​::​​value_type is an unsigned integer
//            type capable of accommodating 32-bit quantities.

// UNSUPPORTED: c++03
// REQUIRES: stdlib=libc++

#include <random>
#include <climits>

#include "test_macros.h"

void f() {
  std::seed_seq seq;

  // Not an integral type
  {
    double* p = nullptr;
    seq.generate(p, p); // expected-error-re@*:* {{static assertion failed{{.+}}: [rand.util.seedseq]/7 requires the value_type of the iterator {{.+}}}}
    // expected-error@*:* 0+ {{invalid operands to}}
  }

  // Not an unsigned type
  {
    long long* p = nullptr;
    seq.generate(p, p); // expected-error-re@*:* {{static assertion failed{{.+}}: [rand.util.seedseq]/7 requires the value_type of the iterator {{.+}}}}
  }

  // Not a 32-bit type
  {
    static_assert(sizeof(unsigned char) * CHAR_BIT < 32, "the test doesn't work on this platform");
    unsigned char* p = nullptr;
    seq.generate(p, p); // expected-error-re@*:* {{static assertion failed{{.+}}: [rand.util.seedseq]/7 requires the value_type of the iterator {{.+}}}}
  }

  // Everything satisfied
  {
    unsigned long* p = nullptr;
    seq.generate(p, p); // no diagnostic
  }
}
