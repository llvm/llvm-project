//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCXX_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H
#define TEST_LIBCXX_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H

#include <vector>

struct TrackCopyMove {
  mutable int copy_count = 0;
  int move_count         = 0;

  constexpr TrackCopyMove() = default;
  constexpr TrackCopyMove(const TrackCopyMove& other) : copy_count(other.copy_count), move_count(other.move_count) {
    ++copy_count;
    ++other.copy_count;
  }

  constexpr TrackCopyMove(TrackCopyMove&& other) noexcept : copy_count(other.copy_count), move_count(other.move_count) {
    ++move_count;
    ++other.move_count;
  }
  constexpr TrackCopyMove& operator=(const TrackCopyMove& other) {
    ++copy_count;
    ++other.copy_count;
    return *this;
  }
  constexpr TrackCopyMove& operator=(TrackCopyMove&& other) noexcept {
    ++move_count;
    ++other.move_count;
    return *this;
  }
  constexpr bool operator==(const TrackCopyMove&) const { return true; }
  constexpr bool operator<(const TrackCopyMove&) const { return false; }
};

template <class T>
struct NotQuiteSequenceContainer : std::vector<T> {
  // hide the name insert_range
  void insert_range() = delete;
};

#endif // TEST_LIBCXX_CONTAINERS_CONTAINER_ADAPTORS_FLAT_HELPERS_H
