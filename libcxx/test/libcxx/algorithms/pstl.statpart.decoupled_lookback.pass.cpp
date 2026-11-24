//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// REQUIRES: std-at-least-c++17
// UNSUPPORTED: libcpp-has-no-incomplete-pstl

// template <class _Tp> struct __decoupled_lookback_partition;

// template <class _Tp> class __decoupled_lookback;

// template <class _Tp>
// optional<__decoupled_lookback<_Tp>> __make_decoupled_lookback(size_t) noexcept;

// template <class _Tp, class _BinaryOperation>
// _Tp __calculate_inclusive_prefix_of_partition(__decoupled_lookback_partition<_Tp>*, _BinaryOperation);

#include <__pstl/backends/statpart/decoupled_lookback.h>
#include <cassert>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>

template <typename _Tp>
using __decoupled_lookback = std::__pstl::__statpart::__decoupled_lookback<_Tp>;
template <typename _Tp>
using __decoupled_lookback_partition = std::__pstl::__statpart::__decoupled_lookback_partition<_Tp>;
using std::__pstl::__statpart::__calculate_inclusive_prefix_of_partition;
using std::__pstl::__statpart::__make_decoupled_lookback;
constexpr auto status_aggregate = std::__pstl::__statpart::__decoupled_lookback_status_aggregate_available;
constexpr auto status_prefix    = std::__pstl::__statpart::__decoupled_lookback_status_prefix_available;

// Tracks the number of alive instances
struct Counted {
  static int alive;
  int value;
  Counted(int v) : value(v) { ++alive; }
  Counted(const Counted& other) : value(other.value) { ++alive; }
  ~Counted() { --alive; }
};
int Counted::alive = 0;

int main(int, char**) {
  // Test partition publishing
  {
    { // Aggregate first, then the inclusive prefix
      __decoupled_lookback_partition<int> p;
      p.__construct_aggregate(5);
      assert(p.__acquire_available_status() == status_aggregate);
      assert(p.__aggregate() == 5);
      p.__construct_inclusive_prefix(7);
      assert(p.__acquire_available_status() == (status_aggregate | status_prefix));
      assert(p.__aggregate() == 5);
      assert(p.__inclusive_prefix() == 7);
    }
    { // Only the inclusive prefix, as done for the first partition
      __decoupled_lookback_partition<int> p;
      p.__construct_inclusive_prefix(42);
      assert(p.__acquire_available_status() == status_prefix);
      assert(p.__inclusive_prefix() == 42);
    }
    { // The constructor arguments are forwarded
      __decoupled_lookback_partition<std::string> p;
      p.__construct_aggregate(3, 'a');
      p.__construct_inclusive_prefix("bbb");
      assert(p.__aggregate() == "aaa");
      assert(p.__inclusive_prefix() == "bbb");
    }
  }

  // Test partition destruction
  {
    { // Nothing constructed
      {
        __decoupled_lookback_partition<Counted> p;
      }
      assert(Counted::alive == 0);
    }
    { // Only the aggregate
      {
        __decoupled_lookback_partition<Counted> p;
        p.__construct_aggregate(1);
        assert(Counted::alive == 1);
      }
      assert(Counted::alive == 0);
    }
    { // Only the inclusive prefix
      {
        __decoupled_lookback_partition<Counted> p;
        p.__construct_inclusive_prefix(1);
        assert(Counted::alive == 1);
      }
      assert(Counted::alive == 0);
    }
    { // Both values
      {
        __decoupled_lookback_partition<Counted> p;
        p.__construct_aggregate(1);
        p.__construct_inclusive_prefix(2);
        assert(Counted::alive == 2);
      }
      assert(Counted::alive == 0);
    }
    { // The lookback destroys every partition
      {
        auto lookback = __make_decoupled_lookback<Counted>(3);
        assert(lookback.has_value());
        assert(lookback->__size() == 3);
        lookback->__partition(0).__construct_inclusive_prefix(1);
        lookback->__partition(1).__construct_aggregate(2);
        lookback->__partition(1).__construct_inclusive_prefix(3);
        lookback->__partition(2).__construct_aggregate(4);
        assert(Counted::alive == 4);
      }
      assert(Counted::alive == 0);
    }
  }

  // Test lookback storage
  {
    { // Empty
      auto lookback = __make_decoupled_lookback<int>(0);
      assert(lookback.has_value());
      assert(lookback->__size() == 0);
    }
    { // The partitions are contiguous
      auto lookback = __make_decoupled_lookback<std::string>(5);
      assert(lookback.has_value());
      assert(lookback->__size() == 5);
      assert(lookback->__partitions() != nullptr);
      for (std::size_t i = 0; i < lookback->__size(); ++i) {
        auto* p = &lookback->__partition(i);
        assert(p == lookback->__partitions() + i);
      }
    }
  }

  // Test __calculate_inclusive_prefix_of_partition
  {
    auto concat = [](const std::string& lhs, const std::string& rhs) { return lhs + rhs; };
    { // The previous partition has the prefix
      auto lookback = __make_decoupled_lookback<std::string>(2);
      assert(lookback.has_value());
      lookback->__partition(0).__construct_inclusive_prefix("a");
      lookback->__partition(1).__construct_aggregate("b");
      assert(__calculate_inclusive_prefix_of_partition(&lookback->__partition(1), concat) == "ab");
    }
    { // Walks all the way back to the first partition
      auto lookback = __make_decoupled_lookback<std::string>(4);
      assert(lookback.has_value());
      lookback->__partition(0).__construct_inclusive_prefix("a");
      lookback->__partition(1).__construct_aggregate("b");
      lookback->__partition(2).__construct_aggregate("c");
      lookback->__partition(3).__construct_aggregate("d");
      assert(__calculate_inclusive_prefix_of_partition(&lookback->__partition(2), concat) == "abc");
      assert(__calculate_inclusive_prefix_of_partition(&lookback->__partition(3), concat) == "abcd");
    }
    { // Stops at the first partition that has a prefix and doesn't look any further
      auto lookback = __make_decoupled_lookback<std::string>(4);
      assert(lookback.has_value());
      lookback->__partition(0).__construct_inclusive_prefix("a");
      lookback->__partition(1).__construct_aggregate("b");
      lookback->__partition(1).__construct_inclusive_prefix("X");
      lookback->__partition(2).__construct_aggregate("c");
      lookback->__partition(3).__construct_aggregate("d");
      assert(__calculate_inclusive_prefix_of_partition(&lookback->__partition(3), concat) == "Xcd");
    }
    { // The starting partition may have already published its prefix concurrently, its aggregate is still used
      auto lookback = __make_decoupled_lookback<std::string>(3);
      assert(lookback.has_value());
      lookback->__partition(0).__construct_inclusive_prefix("a");
      lookback->__partition(1).__construct_aggregate("b");
      lookback->__partition(2).__construct_aggregate("c");
      lookback->__partition(2).__construct_inclusive_prefix("X");
      assert(__calculate_inclusive_prefix_of_partition(&lookback->__partition(2), concat) == "abc");
    }
  }
  return 0;
}
