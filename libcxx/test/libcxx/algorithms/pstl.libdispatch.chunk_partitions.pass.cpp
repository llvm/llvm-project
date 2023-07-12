//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: libcpp-pstl-cpu-backend-libdispatch

// ADDITIONAL_COMPILE_FLAGS: -Wno-private-header

// __chunk_partitions __partition_chunks(ptrdiff_t);

#include <__algorithm/pstl_backends/cpu_backends/libdispatch.h>
#include <cassert>
#include <cstddef>

int main(int, char**) {
  for (std::ptrdiff_t i = 0; i != 2ll << 20; ++i) {
    auto chunks = std::__par_backend::__libdispatch::__partition_chunks(i);
    assert(chunks.__chunk_count_ <= i);
    assert((chunks.__chunk_count_ - 1) * chunks.__chunk_size_ + chunks.__first_chunk_size_ == i);
  }
  return 0;
}
