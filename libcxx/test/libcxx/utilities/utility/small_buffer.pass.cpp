//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: availability-aligned_allocation-missing

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__utility/small_buffer.h>
TEST_DIAGNOSTIC_POP

#include <cassert>
#include <memory>
#include <utility>

struct NotTriviallyRelocatable {
  char c_;

  NotTriviallyRelocatable(char c) : c_(c) {}
  ~NotTriviallyRelocatable() {}
};

struct alignas(16) Overaligned {
  int i;
};

int main(int, char**) {
  using BufferT = std::__small_buffer<8, 8>;
  static_assert(sizeof(BufferT) == 8);
  static_assert(alignof(BufferT) == 8);
  static_assert(BufferT::__fits_in_buffer<int>);
  static_assert(!BufferT::__fits_in_buffer<Overaligned>);
  static_assert(!BufferT::__fits_in_buffer<NotTriviallyRelocatable>);

  BufferT buf;

  { // construct/destroy in the same place
    buf.__construct<int>(3);
    assert(*buf.__get<int>() == 3);
    std::destroy_at(buf.__get<int>());
    buf.__dealloc<int>();

    buf.__construct<NotTriviallyRelocatable>(3);
    assert(buf.__get<NotTriviallyRelocatable>()->c_ == 3);
    std::destroy_at(buf.__get<NotTriviallyRelocatable>());
    buf.__dealloc<NotTriviallyRelocatable>();
  }

  { // Move the buffer around
    buf.__construct<int>(3);
    assert(*buf.__get<int>() == 3);
    auto buf2 = std::move(buf);
    assert(*buf2.__get<int>() == 3);
    std::destroy_at(buf2.__get<int>());
    buf2.__dealloc<int>();

    buf.__construct<NotTriviallyRelocatable>(3);
    assert(buf.__get<NotTriviallyRelocatable>()->c_ == 3);
    auto buf3 = std::move(buf);
    assert(buf3.__get<NotTriviallyRelocatable>()->c_ == 3);
    std::destroy_at(buf3.__get<NotTriviallyRelocatable>());
    buf3.__dealloc<NotTriviallyRelocatable>();
  }

  return 0;
}
