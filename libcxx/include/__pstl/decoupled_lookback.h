//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_DECOUPLED_LOOKBACK_H
#define _LIBCPP___PSTL_DECOUPLED_LOOKBACK_H

#include <__atomic/atomic.h>
#include <__atomic/atomic_sync.h>
#include <__config>
#include <__memory/construct_at.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

// Based on the paper "Single-pass Parallel Prefix Scan with Decoupled Look-back" (Merrill, Garland, 2016).

inline constexpr unsigned char __decoupled_lookback_status_invalid             = 0;
inline constexpr unsigned char __decoupled_lookback_status_aggregate_available = 1 << 0;
inline constexpr unsigned char __decoupled_lookback_status_prefix_available    = 1 << 1;

template <typename _Tp>
struct _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition {
  __decoupled_lookback_partition()                                                 = default;
  __decoupled_lookback_partition(const __decoupled_lookback_partition&)            = delete;
  __decoupled_lookback_partition& operator=(const __decoupled_lookback_partition&) = delete;

  // Destructor: destroy the aggregate and inclusive prefix if they have been constructed.
  ~__decoupled_lookback_partition() {
    unsigned char __flag = __status_flag.load(std::memory_order_relaxed);
    if (__flag & __decoupled_lookback_status_aggregate_available) {
      __aggregate().~_Tp();
    }
    if (__flag & __decoupled_lookback_status_prefix_available) {
      __inclusive_prefix().~_Tp();
    }
  }

  // Acquire the current status of the partition.
  // If no value is published is available yet, wait until one becomes available.
  _LIBCPP_HIDE_FROM_ABI unsigned char __acquire_available_status() {
    unsigned char __current_status = __status_flag.load(std::memory_order_acquire);
    if (__current_status != __decoupled_lookback_status_invalid)
      return __current_status;
    std::__atomic_wait(__status_flag, __decoupled_lookback_status_invalid, std::memory_order_acquire);
    return __status_flag.load(std::memory_order_acquire);
  }

  // Construct the aggregate value of the partition and publish this change.
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct_aggregate(_Args&&... __args) {
    std::__construct_at(&reinterpret_cast<_Tp&>(__aggregate_storage), std::forward<_Args>(__args)...);
    __status_flag.store(__status_flag.load(std::memory_order_relaxed) | __decoupled_lookback_status_aggregate_available,
                        std::memory_order_release);
    std::__atomic_notify_all(__status_flag);
  }

  // Access the aggregate value of the partition.
  // Precondition: __acquire_available_status() & __decoupled_lookback_status_aggregate_available
  _LIBCPP_HIDE_FROM_ABI const _Tp& __aggregate() const { return reinterpret_cast<const _Tp&>(__aggregate_storage); }

  // Construct the inclusive prefix of the partition and publish this change.
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct_inclusive_prefix(_Args&&... __args) {
    std::__construct_at(&reinterpret_cast<_Tp&>(__inclusive_prefix_storage), std::forward<_Args>(__args)...);
    __status_flag.store(__status_flag.load(std::memory_order_relaxed) | __decoupled_lookback_status_prefix_available,
                        std::memory_order_release);
    std::__atomic_notify_all(__status_flag);
  }

  // Access the inclusive prefix of the partition.
  // Precondition: __acquire_available_status() & __decoupled_lookback_status_prefix_available
  _LIBCPP_HIDE_FROM_ABI const _Tp& __inclusive_prefix() const {
    return reinterpret_cast<const _Tp&>(__inclusive_prefix_storage);
  }

private:
  // Atomic/waitable flag indicating the status of the partition.
  std::atomic<unsigned char> __status_flag{__decoupled_lookback_status_invalid};

  // Storage for the aggregate reduced value of the partition.
  alignas(_Tp) unsigned char __aggregate_storage[sizeof(_Tp)];

  // Storage for the inclusive prefix of the partition.
  alignas(_Tp) unsigned char __inclusive_prefix_storage[sizeof(_Tp)];
};

template <typename _Tp>
class _LIBCPP_HIDE_FROM_ABI __decoupled_lookback {
public:
  // Allocates the memory for the lookback partitions.
  // Does not throw, instead sets the size to 0 if allocation fails.
  _LIBCPP_HIDE_FROM_ABI explicit __decoupled_lookback(size_t __size) {
    __partitions_       = __size > 0 ? new (std::nothrow) __decoupled_lookback_partition<_Tp>[__size] : nullptr;
    __partitions_count_ = __partitions_ ? __size : 0;
  }

  _LIBCPP_HIDE_FROM_ABI ~__decoupled_lookback() { delete[] __partitions_; }

  __decoupled_lookback(const __decoupled_lookback&) = delete;

  __decoupled_lookback& operator=(const __decoupled_lookback&) = delete;

  _LIBCPP_HIDE_FROM_ABI size_t __size() const { return __partitions_count_; }

  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition<_Tp>* __partitions() const { return __partitions_; }

  _LIBCPP_HIDE_FROM_ABI __decoupled_lookback_partition<_Tp>& __partition(size_t __index) const {
    return __partitions_[__index];
  }

  // Calculate an exclusive prefix starting at the given partition.
  // Performs reductions of aggregate values until it arrives at a partition with an available prefix.
  // Precondition: __partition(__index).status is __decoupled_lookback_status_aggregate_available.
  template <class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI _Tp __calculate_exclusive_prefix(size_t __index, _BinaryOperation __reduce) {
    // Start at the given partition
    _Tp __prefix = __partitions_[__index].__aggregate();
    while (true) {
      // Move to the previous partition and check its status.
      --__index;
      __decoupled_lookback_partition< _Tp >& __prev_partition = __partitions_[__index];
      unsigned char __status                                  = __prev_partition.__acquire_available_status();
      if (__status & __decoupled_lookback_status_prefix_available) {
        // Found a partition with an available prefix - can perform a final reduction and terminate.
        __prefix = __reduce(__prev_partition.__inclusive_prefix(), std::move(__prefix));
        break;
      } else /* if(__status & __decoupled_lookback_status_aggregate_available) */ {
        // Reduce with another aggregate and continue moving left.
        __prefix = __reduce(__prev_partition.__aggregate(), std::move(__prefix));
      }
    }
    return __prefix;
  }

private:
  __decoupled_lookback_partition<_Tp>* __partitions_;
  size_t __partitions_count_;
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_DECOUPLED_LOOKBACK_H
