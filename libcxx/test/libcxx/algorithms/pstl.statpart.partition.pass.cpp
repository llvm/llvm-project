//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17
// REQUIRES: libcpp-pstl-backend-statpart
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class Iterator>
// __partitioning_result<Iterator>
// __partition(Iterator __first,
//             Iterator __last,
//             size_t __min_partition_size,
//             size_t __prefix_to_skip,
//             size_t __suffix_to_skip,
//             size_t __workers_count)

#include <__pstl/backends/statpart.h>
#include <algorithm>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <vector>

template <class Container>
void test_partition() {
  auto part_eq = [](const auto& partition, const auto& container, size_t first_pos, size_t last_pos) {
    return partition.__first == std::next(container.begin(), first_pos) &&
           partition.__last == std::next(container.begin(), last_pos);
  };

  { // 1 element, 8 workers: no partitions
    Container v(1);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 1, 0, 0, 8);
    assert(partitions.__value_ == std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__not_enough_elements);
  }
  { // 2 elements, 8 workers: 2 partitions
    // [0..1), [1..2)
    Container v(2);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 1, 0, 0, 8);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 2);
    assert(partitions.__value_->__end() == v.end());
    assert(part_eq(partitions.__value_->__partition(0), v, 0, 1));
    assert(part_eq(partitions.__value_->__partition(1), v, 1, 2));
  }
  { // 2 elements, min_size=2, 8 workers: no partitions
    // [0..1), [1..2)
    Container v(2);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 1, 2, 0, 8);
    assert(partitions.__value_ == std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__not_enough_elements);
  }
  { // 20 elements, 2 workers: 16 partitions
    // [0..2), [2..4), [4..6), [6..8), [8..9), [9..10), ..., [19, 20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 1, 0, 0, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 16);
    assert(partitions.__value_->__end() == v.end());
    assert(part_eq(partitions.__value_->__partition(0), v, 0, 2));
    assert(part_eq(partitions.__value_->__partition(1), v, 2, 4));
    assert(part_eq(partitions.__value_->__partition(2), v, 4, 6));
    assert(part_eq(partitions.__value_->__partition(3), v, 6, 8));
    assert(part_eq(partitions.__value_->__partition(4), v, 8, 9));
    assert(part_eq(partitions.__value_->__partition(5), v, 9, 10));
    assert(part_eq(partitions.__value_->__partition(6), v, 10, 11));
    assert(part_eq(partitions.__value_->__partition(7), v, 11, 12));
    assert(part_eq(partitions.__value_->__partition(8), v, 12, 13));
    assert(part_eq(partitions.__value_->__partition(9), v, 13, 14));
    assert(part_eq(partitions.__value_->__partition(10), v, 14, 15));
    assert(part_eq(partitions.__value_->__partition(11), v, 15, 16));
    assert(part_eq(partitions.__value_->__partition(12), v, 16, 17));
    assert(part_eq(partitions.__value_->__partition(13), v, 17, 18));
    assert(part_eq(partitions.__value_->__partition(14), v, 18, 19));
    assert(part_eq(partitions.__value_->__partition(15), v, 19, 20));
  }
  { // 20 elements, min_size=2, 2 workers: 10 partitions
    // [0..2), [2..4), [4..6), ..., [18, 20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 2, 0, 0, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 10);
    assert(partitions.__value_->__end() == v.end());
    assert(part_eq(partitions.__value_->__partition(0), v, 0, 2));
    assert(part_eq(partitions.__value_->__partition(1), v, 2, 4));
    assert(part_eq(partitions.__value_->__partition(2), v, 4, 6));
    assert(part_eq(partitions.__value_->__partition(3), v, 6, 8));
    assert(part_eq(partitions.__value_->__partition(4), v, 8, 10));
    assert(part_eq(partitions.__value_->__partition(5), v, 10, 12));
    assert(part_eq(partitions.__value_->__partition(6), v, 12, 14));
    assert(part_eq(partitions.__value_->__partition(7), v, 14, 16));
    assert(part_eq(partitions.__value_->__partition(8), v, 16, 18));
    assert(part_eq(partitions.__value_->__partition(9), v, 18, 20));
  }
  { // 20 elements, min_size=3, 2 workers: 6 partitions
    // [0..4), [4..8), [8..11), [11..14), [14..17), [17..20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 3, 0, 0, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 6);
    assert(partitions.__value_->__end() == v.end());
    assert(part_eq(partitions.__value_->__partition(0), v, 0, 4));
    assert(part_eq(partitions.__value_->__partition(1), v, 4, 8));
    assert(part_eq(partitions.__value_->__partition(2), v, 8, 11));
    assert(part_eq(partitions.__value_->__partition(3), v, 11, 14));
    assert(part_eq(partitions.__value_->__partition(4), v, 14, 17));
    assert(part_eq(partitions.__value_->__partition(5), v, 17, 20));
  }
  { // 20 elements, min_size=3, prefix_skip=1, 2 workers: 6 partitions
    // [1..5), [5..8), [8..11), [11..14), [14..17), [17..20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 3, 1, 0, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 6);
    assert(partitions.__value_->__end() == v.end());
    assert(part_eq(partitions.__value_->__partition(0), v, 1, 5));
    assert(part_eq(partitions.__value_->__partition(1), v, 5, 8));
    assert(part_eq(partitions.__value_->__partition(2), v, 8, 11));
    assert(part_eq(partitions.__value_->__partition(3), v, 11, 14));
    assert(part_eq(partitions.__value_->__partition(4), v, 14, 17));
    assert(part_eq(partitions.__value_->__partition(5), v, 17, 20));
  }
  { // 20 elements, min_size=3, suffix_skip=1, 2 workers: 6 partitions
    // [0..4), [4..8), [8..11), [11..14), [14..17), [17..20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 3, 0, 1, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 6);
    assert(partitions.__value_->__end() == std::prev(v.end(), 1));
    assert(part_eq(partitions.__value_->__partition(0), v, 0, 4));
    assert(part_eq(partitions.__value_->__partition(1), v, 4, 7));
    assert(part_eq(partitions.__value_->__partition(2), v, 7, 10));
    assert(part_eq(partitions.__value_->__partition(3), v, 10, 13));
    assert(part_eq(partitions.__value_->__partition(4), v, 13, 16));
    assert(part_eq(partitions.__value_->__partition(5), v, 16, 19));
  }
  { // 20 elements, min_size=3, prefix_skip=1, suffix_skip=1, 2 workers: 6 partitions
    // [0..4), [4..8), [8..11), [11..14), [14..17), [17..20)
    Container v(20);
    const auto partitions = std::__pstl::__statpart::__partition(v.begin(), v.end(), 3, 1, 1, 2);
    assert(partitions.__value_ != std::nullopt);
    assert(partitions.__status_ == std::__pstl::__statpart::__partitioning_status::__ok);
    assert(partitions.__value_->__partitions_count() == 6);
    assert(partitions.__value_->__end() == std::prev(v.end(), 1));
    assert(part_eq(partitions.__value_->__partition(0), v, 1, 4));
    assert(part_eq(partitions.__value_->__partition(1), v, 4, 7));
    assert(part_eq(partitions.__value_->__partition(2), v, 7, 10));
    assert(part_eq(partitions.__value_->__partition(3), v, 10, 13));
    assert(part_eq(partitions.__value_->__partition(4), v, 13, 16));
    assert(part_eq(partitions.__value_->__partition(5), v, 16, 19));
  }
}

int main(int, char**) {
  test_partition<std::vector<int>>();
  // later: test_partition<std::list<int>>();
}
