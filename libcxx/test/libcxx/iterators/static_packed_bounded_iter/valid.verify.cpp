//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template <class _Ptr, size_t _RangeCapacity>
// class __static_packed_bounded_iterator;

#include <__iterator/static_packed_bounded_iter.h>
#include <cstddef>
#include <cstdint>

#include "test_iterators.h"

void test() {
  std::__static_packed_bounded_iterator<std::int8_t*, 1>
      v1; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires the range to fit in the alignment bits}}
  std::__static_packed_bounded_iterator<std::int16_t*, 1>
      v2; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires the range to fit in the alignment bits}}
  std::__static_packed_bounded_iterator<std::int32_t*, 3>
      v3; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires the range to fit in the alignment bits}}
  std::__static_packed_bounded_iterator<std::int64_t*, 7>
      v4; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires the range to fit in the alignment bits}}

  std::__static_packed_bounded_iterator<cpp20_random_access_iterator<int*>, 0>
      v5; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires a pointer type}}
  std::__static_packed_bounded_iterator<contiguous_iterator<int*>, 0>
      v6; // expected-error-re@*:* {{static assertion failed{{.*}}: __static_packed_bounded_iterator requires a pointer type}}
}
