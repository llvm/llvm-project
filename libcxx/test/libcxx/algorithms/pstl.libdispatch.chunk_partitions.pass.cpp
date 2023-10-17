//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: libcpp-pstl-cpu-backend-libdispatch

// __chunk_partitions __partition_chunks(ptrdiff_t);

#include <__algorithm/pstl_backends/cpu_backends/libdispatch.h>
#include <cassert>
#include <cstddef>

int main(int, char**) {
  {
    auto chunks = std::__par_backend::__libdispatch::__partition_chunks(0);
    assert(chunks.__chunk_count_ == 1);
    assert(chunks.__first_chunk_size_ == 0);
    assert(chunks.__chunk_size_ == 0);
  }

  {
    auto chunks = std::__par_backend::__libdispatch::__partition_chunks(1);
    assert(chunks.__chunk_count_ == 1);
    assert(chunks.__first_chunk_size_ == 1);
    assert(chunks.__chunk_size_ == 1);
  }

  for (std::ptrdiff_t i = 2; i != 2ll << 20; ++i) {
    auto chunks = std::__par_backend::__libdispatch::__partition_chunks(i);
    assert(chunks.__chunk_count_ >= 1);
    assert(chunks.__chunk_count_ <= i);
    assert((chunks.__chunk_count_ - 1) * chunks.__chunk_size_ + chunks.__first_chunk_size_ == i);
  }
  return 0;
}
