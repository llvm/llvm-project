//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://github.com/llvm/llvm-project/issues/40340 is fixed
// UNSUPPORTED: availability-pmr-missing

// <memory_resource>

// struct pool_options
// class unsynchronized_pool_resource
// class synchronized_pool_resource

#include <memory_resource>
#include <cassert>
#include <cstdint> // SIZE_MAX, UINT32_MAX

static void assert_options(const std::pmr::pool_options& actual, const std::pmr::pool_options& expected) {
  assert(actual.max_blocks_per_chunk == expected.max_blocks_per_chunk);
  assert(actual.largest_required_pool_block == expected.largest_required_pool_block);
}

void test_pool_options(std::pmr::pool_options initial, std::pmr::pool_options expected) {
  std::pmr::unsynchronized_pool_resource mr(initial, std::pmr::null_memory_resource());
  assert_options(mr.options(), expected);

  std::pmr::synchronized_pool_resource mr2(initial, std::pmr::null_memory_resource());
  assert_options(mr2.options(), expected);
}

int main(int, char**) {
  test_pool_options({0, 0}, {1048576, 1048576});
  test_pool_options({0, 1}, {1048576, 8});
  test_pool_options({0, 2}, {1048576, 8});
  test_pool_options({0, 4}, {1048576, 8});
  test_pool_options({0, 8}, {1048576, 8});
  test_pool_options({0, 16}, {1048576, 16});
  test_pool_options({0, 32}, {1048576, 32});
  test_pool_options({0, 1024}, {1048576, 1024});
  test_pool_options({0, 1048576}, {1048576, 1048576});
  test_pool_options({0, 2097152}, {1048576, 2097152});
  test_pool_options({0, 1073741824}, {1048576, 1073741824});
  test_pool_options({0, 2147483648}, {1048576, 1073741824});
  test_pool_options({1, 0}, {16, 1048576});
  test_pool_options({1, 1}, {16, 8});
  test_pool_options({1, 2}, {16, 8});
  test_pool_options({1, 4}, {16, 8});
  test_pool_options({1, 8}, {16, 8});
  test_pool_options({1, 16}, {16, 16});
  test_pool_options({1, 32}, {16, 32});
  test_pool_options({1, 1024}, {16, 1024});
  test_pool_options({1, 1048576}, {16, 1048576});
  test_pool_options({1, 2097152}, {16, 2097152});
  test_pool_options({1, 1073741824}, {16, 1073741824});
  test_pool_options({1, 2147483648}, {16, 1073741824});
  test_pool_options({2, 0}, {16, 1048576});
  test_pool_options({2, 1}, {16, 8});
  test_pool_options({2, 2}, {16, 8});
  test_pool_options({2, 4}, {16, 8});
  test_pool_options({2, 8}, {16, 8});
  test_pool_options({2, 16}, {16, 16});
  test_pool_options({2, 32}, {16, 32});
  test_pool_options({2, 1024}, {16, 1024});
  test_pool_options({2, 1048576}, {16, 1048576});
  test_pool_options({2, 2097152}, {16, 2097152});
  test_pool_options({2, 1073741824}, {16, 1073741824});
  test_pool_options({2, 2147483648}, {16, 1073741824});
  test_pool_options({4, 0}, {16, 1048576});
  test_pool_options({4, 1}, {16, 8});
  test_pool_options({4, 2}, {16, 8});
  test_pool_options({4, 4}, {16, 8});
  test_pool_options({4, 8}, {16, 8});
  test_pool_options({4, 16}, {16, 16});
  test_pool_options({4, 32}, {16, 32});
  test_pool_options({4, 1024}, {16, 1024});
  test_pool_options({4, 1048576}, {16, 1048576});
  test_pool_options({4, 2097152}, {16, 2097152});
  test_pool_options({4, 1073741824}, {16, 1073741824});
  test_pool_options({4, 2147483648}, {16, 1073741824});
  test_pool_options({8, 0}, {16, 1048576});
  test_pool_options({8, 1}, {16, 8});
  test_pool_options({8, 2}, {16, 8});
  test_pool_options({8, 4}, {16, 8});
  test_pool_options({8, 8}, {16, 8});
  test_pool_options({8, 16}, {16, 16});
  test_pool_options({8, 32}, {16, 32});
  test_pool_options({8, 1024}, {16, 1024});
  test_pool_options({8, 1048576}, {16, 1048576});
  test_pool_options({8, 2097152}, {16, 2097152});
  test_pool_options({8, 1073741824}, {16, 1073741824});
  test_pool_options({8, 2147483648}, {16, 1073741824});
  test_pool_options({16, 0}, {16, 1048576});
  test_pool_options({16, 1}, {16, 8});
  test_pool_options({16, 2}, {16, 8});
  test_pool_options({16, 4}, {16, 8});
  test_pool_options({16, 8}, {16, 8});
  test_pool_options({16, 16}, {16, 16});
  test_pool_options({16, 32}, {16, 32});
  test_pool_options({16, 1024}, {16, 1024});
  test_pool_options({16, 1048576}, {16, 1048576});
  test_pool_options({16, 2097152}, {16, 2097152});
  test_pool_options({16, 1073741824}, {16, 1073741824});
  test_pool_options({16, 2147483648}, {16, 1073741824});
  test_pool_options({32, 0}, {32, 1048576});
  test_pool_options({32, 1}, {32, 8});
  test_pool_options({32, 2}, {32, 8});
  test_pool_options({32, 4}, {32, 8});
  test_pool_options({32, 8}, {32, 8});
  test_pool_options({32, 16}, {32, 16});
  test_pool_options({32, 32}, {32, 32});
  test_pool_options({32, 1024}, {32, 1024});
  test_pool_options({32, 1048576}, {32, 1048576});
  test_pool_options({32, 2097152}, {32, 2097152});
  test_pool_options({32, 1073741824}, {32, 1073741824});
  test_pool_options({32, 2147483648}, {32, 1073741824});
  test_pool_options({1024, 0}, {1024, 1048576});
  test_pool_options({1024, 1}, {1024, 8});
  test_pool_options({1024, 2}, {1024, 8});
  test_pool_options({1024, 4}, {1024, 8});
  test_pool_options({1024, 8}, {1024, 8});
  test_pool_options({1024, 16}, {1024, 16});
  test_pool_options({1024, 32}, {1024, 32});
  test_pool_options({1024, 1024}, {1024, 1024});
  test_pool_options({1024, 1048576}, {1024, 1048576});
  test_pool_options({1024, 2097152}, {1024, 2097152});
  test_pool_options({1024, 1073741824}, {1024, 1073741824});
  test_pool_options({1024, 2147483648}, {1024, 1073741824});
  test_pool_options({1048576, 0}, {1048576, 1048576});
  test_pool_options({1048576, 1}, {1048576, 8});
  test_pool_options({1048576, 2}, {1048576, 8});
  test_pool_options({1048576, 4}, {1048576, 8});
  test_pool_options({1048576, 8}, {1048576, 8});
  test_pool_options({1048576, 16}, {1048576, 16});
  test_pool_options({1048576, 32}, {1048576, 32});
  test_pool_options({1048576, 1024}, {1048576, 1024});
  test_pool_options({1048576, 1048576}, {1048576, 1048576});
  test_pool_options({1048576, 2097152}, {1048576, 2097152});
  test_pool_options({1048576, 1073741824}, {1048576, 1073741824});
  test_pool_options({1048576, 2147483648}, {1048576, 1073741824});
  test_pool_options({2097152, 0}, {1048576, 1048576});
  test_pool_options({2097152, 1}, {1048576, 8});
  test_pool_options({2097152, 2}, {1048576, 8});
  test_pool_options({2097152, 4}, {1048576, 8});
  test_pool_options({2097152, 8}, {1048576, 8});
  test_pool_options({2097152, 16}, {1048576, 16});
  test_pool_options({2097152, 32}, {1048576, 32});
  test_pool_options({2097152, 1024}, {1048576, 1024});
  test_pool_options({2097152, 1048576}, {1048576, 1048576});
  test_pool_options({2097152, 2097152}, {1048576, 2097152});
  test_pool_options({2097152, 1073741824}, {1048576, 1073741824});
  test_pool_options({2097152, 2147483648}, {1048576, 1073741824});
  test_pool_options({1073741824, 0}, {1048576, 1048576});
  test_pool_options({1073741824, 1}, {1048576, 8});
  test_pool_options({1073741824, 2}, {1048576, 8});
  test_pool_options({1073741824, 4}, {1048576, 8});
  test_pool_options({1073741824, 8}, {1048576, 8});
  test_pool_options({1073741824, 16}, {1048576, 16});
  test_pool_options({1073741824, 32}, {1048576, 32});
  test_pool_options({1073741824, 1024}, {1048576, 1024});
  test_pool_options({1073741824, 1048576}, {1048576, 1048576});
  test_pool_options({1073741824, 2097152}, {1048576, 2097152});
  test_pool_options({1073741824, 1073741824}, {1048576, 1073741824});
  test_pool_options({1073741824, 2147483648}, {1048576, 1073741824});
  test_pool_options({2147483648, 0}, {1048576, 1048576});
  test_pool_options({2147483648, 1}, {1048576, 8});
  test_pool_options({2147483648, 2}, {1048576, 8});
  test_pool_options({2147483648, 4}, {1048576, 8});
  test_pool_options({2147483648, 8}, {1048576, 8});
  test_pool_options({2147483648, 16}, {1048576, 16});
  test_pool_options({2147483648, 32}, {1048576, 32});
  test_pool_options({2147483648, 1024}, {1048576, 1024});
  test_pool_options({2147483648, 1048576}, {1048576, 1048576});
  test_pool_options({2147483648, 2097152}, {1048576, 2097152});
  test_pool_options({2147483648, 1073741824}, {1048576, 1073741824});
  test_pool_options({2147483648, 2147483648}, {1048576, 1073741824});

#if SIZE_MAX > UINT32_MAX
  test_pool_options({0, 8589934592}, {1048576, 1073741824});
  test_pool_options({1, 8589934592}, {16, 1073741824});
  test_pool_options({2, 8589934592}, {16, 1073741824});
  test_pool_options({4, 8589934592}, {16, 1073741824});
  test_pool_options({8, 8589934592}, {16, 1073741824});
  test_pool_options({16, 8589934592}, {16, 1073741824});
  test_pool_options({32, 8589934592}, {32, 1073741824});
  test_pool_options({1024, 8589934592}, {1024, 1073741824});
  test_pool_options({1048576, 8589934592}, {1048576, 1073741824});
  test_pool_options({2097152, 8589934592}, {1048576, 1073741824});
  test_pool_options({1073741824, 8589934592}, {1048576, 1073741824});
  test_pool_options({2147483648, 8589934592}, {1048576, 1073741824});
  test_pool_options({8589934592, 0}, {1048576, 1048576});
  test_pool_options({8589934592, 1}, {1048576, 8});
  test_pool_options({8589934592, 2}, {1048576, 8});
  test_pool_options({8589934592, 4}, {1048576, 8});
  test_pool_options({8589934592, 8}, {1048576, 8});
  test_pool_options({8589934592, 16}, {1048576, 16});
  test_pool_options({8589934592, 32}, {1048576, 32});
  test_pool_options({8589934592, 1024}, {1048576, 1024});
  test_pool_options({8589934592, 1048576}, {1048576, 1048576});
  test_pool_options({8589934592, 2097152}, {1048576, 2097152});
  test_pool_options({8589934592, 1073741824}, {1048576, 1073741824});
  test_pool_options({8589934592, 2147483648}, {1048576, 1073741824});
  test_pool_options({8589934592, 8589934592}, {1048576, 1073741824});
#endif

  return 0;
}
