//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_DECOUPLED_LOOKBACK_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_DECOUPLED_LOOKBACK_H

#include <__assert>
#include <__atomic/atomic.h>
#include <__atomic/atomic_sync.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/addressof.h>
#include <__memory/construct_at.h>
#include <__new/global_new_delete.h>
#include <__new/nothrow_t.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__utility/exchange.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

namespace __statpart {

// Based on the paper "Single-pass Parallel Prefix Scan with Decoupled Look-back" (Merrill, Garland, 2016).

inline constexpr unsigned char __decoupled_lookback_status_invalid             = 0;
inline constexpr unsigned char __decoupled_lookback_status_aggregate_available = 1 << 0;
inline constexpr unsigned char __decoupled_lookback_status_prefix_available    = 1 << 1;

// __decoupled_lookback_partition contains 2 values of _Tp (local aggregate and inclusive prefix) and a status flag.
// The values can be initialized by a thread working on this partition by calling __construct_aggregate or
// __construct_inclusive_prefix. These functions implicitly update the status flag and notify any thread potentially
// waiting on it.
// Other threads reaching this partition via a lookback should use __acquire_available_status() to wait until either of
// the two values is available. Depending on the status flag, the constructed values can be accessed via __aggregate()
// and __inclusive_prefix().
template <typename _Tp>
struct _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition {
  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition() {}
  __decoupled_lookback_partition(const __decoupled_lookback_partition&)            = delete;
  __decoupled_lookback_partition& operator=(const __decoupled_lookback_partition&) = delete;

  // Destructor: destroy the aggregate and inclusive prefix if they have been constructed.
  _LIBCPP_HIDE_FROM_ABI ~__decoupled_lookback_partition() {
    unsigned char __flag = __status_flag_.load(std::memory_order_relaxed);
    if (__flag & __decoupled_lookback_status_aggregate_available) {
      __aggregate_.~_Tp();
    }
    if (__flag & __decoupled_lookback_status_prefix_available) {
      __inclusive_prefix_.~_Tp();
    }
  }

  // Acquire the current status of the partition.
  // If no value is published yet, wait until one becomes available.
  _LIBCPP_HIDE_FROM_ABI unsigned char __acquire_available_status() {
    unsigned char __current_status = __status_flag_.load(std::memory_order_acquire);
    if (__current_status != __decoupled_lookback_status_invalid)
      return __current_status;
    std::__atomic_wait(__status_flag_, __decoupled_lookback_status_invalid, std::memory_order_acquire);
    return __status_flag_.load(std::memory_order_acquire);
  }

  // Construct the aggregate value of the partition and publish this change.
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct_aggregate(_Args&&... __args) {
    _LIBCPP_ASSERT_INTERNAL((__status_flag_.load() & __decoupled_lookback_status_aggregate_available) == 0,
                            "__decoupled_lookback_status_aggregate_available flag must not be set");
    std::__construct_at(std::addressof(__aggregate_), std::forward<_Args>(__args)...);
    __status_flag_.fetch_or(__decoupled_lookback_status_aggregate_available, std::memory_order_release);
    std::__atomic_notify_all(__status_flag_);
  }

  // Access the aggregate value of the partition.
  // Precondition: __acquire_available_status() & __decoupled_lookback_status_aggregate_available
  _LIBCPP_HIDE_FROM_ABI const _Tp& __aggregate() const {
    _LIBCPP_ASSERT_INTERNAL((__status_flag_.load() & __decoupled_lookback_status_aggregate_available) != 0,
                            "__decoupled_lookback_status_aggregate_available flag must be set");
    return __aggregate_;
  }

  // Construct the inclusive prefix of the partition and publish this change.
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct_inclusive_prefix(_Args&&... __args) {
    _LIBCPP_ASSERT_INTERNAL((__status_flag_.load() & __decoupled_lookback_status_prefix_available) == 0,
                            "__decoupled_lookback_status_prefix_available flag must not be set");
    std::__construct_at(std::addressof(__inclusive_prefix_), std::forward<_Args>(__args)...);
    __status_flag_.fetch_or(__decoupled_lookback_status_prefix_available, std::memory_order_release);
    std::__atomic_notify_all(__status_flag_);
  }

  // Access the inclusive prefix of the partition.
  // Precondition: __acquire_available_status() & __decoupled_lookback_status_prefix_available
  _LIBCPP_HIDE_FROM_ABI const _Tp& __inclusive_prefix() const {
    _LIBCPP_ASSERT_INTERNAL((__status_flag_.load() & __decoupled_lookback_status_prefix_available) != 0,
                            "__decoupled_lookback_status_prefix_available flag must be set");
    return __inclusive_prefix_;
  }

private:
  // Storage for the aggregate reduced value of the partition.
  union {
    _Tp __aggregate_;
  };

  // Storage for the inclusive prefix of the partition.
  union {
    _Tp __inclusive_prefix_;
  };

  // Atomic/waitable flag indicating the status of the partition.
  std::atomic<unsigned char> __status_flag_{__decoupled_lookback_status_invalid};

  // NB! Currently the layout of __decoupled_lookback_partition allows for false sharing to occur.
};

// __decoupled_lookback holds a dynamically-allocated array of __decoupled_lookback_partition.
// Created by __make_decoupled_lookback().
template <typename _Tp>
class _LIBCPP_HIDE_FROM_ABI __decoupled_lookback {
public:
  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback(__decoupled_lookback&& __other) noexcept
      : __partitions_(std::exchange(__other.__partitions_, nullptr)),
        __partitions_count_(std::exchange(__other.__partitions_count_, 0)) {}

  __decoupled_lookback(const __decoupled_lookback&)            = delete;
  __decoupled_lookback& operator=(const __decoupled_lookback&) = delete;
  __decoupled_lookback& operator=(__decoupled_lookback&&)      = delete;

  _LIBCPP_HIDE_FROM_ABI ~__decoupled_lookback() { delete[] __partitions_; }

  // Returns the number of partitions in the lookback.
  _LIBCPP_HIDE_FROM_ABI size_t __size() const { return __partitions_count_; }

  // Provides a pointer to the first partition in a contiguous storage.
  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition<_Tp>* __partitions() const { return __partitions_; }

  // Provides a reference to the specified partition.
  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition<_Tp>& __partition(size_t __index) const {
    _LIBCPP_ASSERT_INTERNAL(__index < __partitions_count_, "__index must be less than __partitions_count_");
    return __partitions_[__index];
  }

private:
  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback() = default;

  __decoupled_lookback_partition<_Tp>* __partitions_ = nullptr;
  size_t __partitions_count_                         = 0;

  template <class _Up>
  friend _LIBCPP_HIDE_FROM_ABI optional<__decoupled_lookback<_Up>> __make_decoupled_lookback(size_t) noexcept;
};

// Creates a lookback partition with the specified capacity.
// In case memory allocation fails, returns std::nullopt.
template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI optional<__decoupled_lookback<_Tp>> __make_decoupled_lookback(size_t __size) noexcept {
  __decoupled_lookback<_Tp> __lookback;
  __lookback.__partitions_ = new (std::nothrow) __decoupled_lookback_partition<_Tp>[__size];
  if (!__lookback.__partitions_)
    return nullopt;
  __lookback.__partitions_count_ = __size;
  return __lookback;
}

// Calculate an inclusive prefix of a given partition that currently has only an aggregate.
// Performs reductions of aggregate values until it arrives at a partition with an available prefix.
// Precondition: __partition->__acquire_available_status() == __decoupled_lookback_status_aggregate_available.
template <typename _Tp, class _BinaryOperation>
_LIBCPP_HIDE_FROM_ABI _Tp
__calculate_inclusive_prefix_of_partition(__decoupled_lookback_partition<_Tp>* __partition, _BinaryOperation __reduce) {
  // Start at the given partition.
  _Tp __prefix = __partition->__aggregate();
  while (true) {
    // Move to the previous partition and check its status.
    --__partition;
    unsigned char __status = __partition->__acquire_available_status();
    if (__status & __decoupled_lookback_status_prefix_available) {
      // Found a partition with an available prefix - can perform a final reduction and terminate.
      return __reduce(__partition->__inclusive_prefix(), std::move(__prefix));
    }
    // Reduce with another aggregate and continue moving left.
    __prefix = __reduce(__partition->__aggregate(), std::move(__prefix));
  }
}

} // namespace __statpart

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_DECOUPLED_LOOKBACK_H
