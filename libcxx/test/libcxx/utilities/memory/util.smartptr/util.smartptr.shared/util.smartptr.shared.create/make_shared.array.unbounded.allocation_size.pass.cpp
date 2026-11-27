//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: sanitizer-new-delete

// <memory>

// shared_ptr

// template<class T> shared_ptr<T> make_shared(size_t N); // T is U[]
//
// template<class T>
// shared_ptr<T> make_shared(size_t N, const remove_extent_t<T>& u); // T is U[]
//
// template<class T> shared_ptr<T> make_shared_for_overwrite(size_t N); // T is U[]

// Check that we don't over-allocate for unbounded arrays.

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "count_new.h"
#include "test_macros.h"

template <class Array>
void test(std::size_t n) {
  using Element                    = std::remove_extent_t<Array>;
  constexpr std::size_t max_header = 256;
  const std::size_t min_bytes      = n * sizeof(Element);
  const std::size_t max_bytes      = min_bytes + max_header;

  {
    globalMemCounter.reset();
    std::shared_ptr<Array> ptr = std::make_shared<Array>(n);
    DoNotOptimize(ptr.get());
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkLastNewSizeGe(min_bytes));
    assert(globalMemCounter.last_new_size <= max_bytes);
  }
  assert(globalMemCounter.checkOutstandingNewEq(0));

  {
    Element init{};
    globalMemCounter.reset();
    std::shared_ptr<Array> ptr = std::make_shared<Array>(n, init);
    DoNotOptimize(ptr.get());
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkLastNewSizeGe(min_bytes));
    assert(globalMemCounter.last_new_size <= max_bytes);
  }
  assert(globalMemCounter.checkOutstandingNewEq(0));

  {
    globalMemCounter.reset();
    std::shared_ptr<Array> ptr = std::make_shared_for_overwrite<Array>(n);
    DoNotOptimize(ptr.get());
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkLastNewSizeGe(min_bytes));
    assert(globalMemCounter.last_new_size <= max_bytes);
  }
  assert(globalMemCounter.checkOutstandingNewEq(0));
}

int main(int, char**) {
  const std::size_t sizes[] = {0, 1, 1024};
  for (std::size_t n : sizes) {
    test<char[]>(n);
    test<int[]>(n);
    test<char[][4]>(n);
  }

  return 0;
}
