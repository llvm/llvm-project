// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___STOP_TOKEN_STOP_STATE_H
#define _LIBCPP___STOP_TOKEN_STOP_STATE_H

#include <__availability>
#include <__config>
#include <__mutex/mutex.h>
#include <__stop_token/intrusive_list_view.h>
#include <__thread/id.h>
#include <atomic>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_THREADS)

struct __stop_callback_base : __intrusive_node_base<__stop_callback_base> {
  using __callback_fn_t = void(__stop_callback_base*) noexcept;
  _LIBCPP_HIDE_FROM_ABI explicit __stop_callback_base(__callback_fn_t* __callback_fn) : __callback_fn_(__callback_fn) {}

  _LIBCPP_HIDE_FROM_ABI void __invoke() noexcept { __callback_fn_(this); }

  __callback_fn_t* __callback_fn_;
  atomic<bool> __completed_ = false;
  bool* __destroyed_        = nullptr;
};

// stop_token needs to lock with noexcept. mutex::lock can throw.
// wrap it with a while loop and catch all exceptions
class __nothrow_mutex_lock {
  std::mutex& __mutex_;
  bool __is_locked_;

public:
  _LIBCPP_HIDE_FROM_ABI explicit __nothrow_mutex_lock(std::mutex& __mutex) noexcept
      : __mutex_(__mutex), __is_locked_(true) {
    __lock();
  }

  __nothrow_mutex_lock(const __nothrow_mutex_lock&)            = delete;
  __nothrow_mutex_lock(__nothrow_mutex_lock&&)                 = delete;
  __nothrow_mutex_lock& operator=(const __nothrow_mutex_lock&) = delete;
  __nothrow_mutex_lock& operator=(__nothrow_mutex_lock&&)      = delete;

  _LIBCPP_HIDE_FROM_ABI ~__nothrow_mutex_lock() {
    if (__is_locked_) {
      __unlock();
    }
  }

  _LIBCPP_HIDE_FROM_ABI bool __owns_lock() const noexcept { return __is_locked_; }

  _LIBCPP_HIDE_FROM_ABI void __lock() noexcept {
    while (true) {
      try {
        __mutex_.lock();
        break;
      } catch (...) {
      }
    }
    __is_locked_ = true;
  }

  _LIBCPP_HIDE_FROM_ABI void __unlock() noexcept {
    __mutex_.unlock(); // throws nothing
    __is_locked_ = false;
  }
};

class __stop_state {
  static constexpr uint32_t __stop_requested_bit        = 1;
  static constexpr uint32_t __stop_source_counter_shift = 1;

  // The "stop_source counter" is not used for lifetime reference counting.
  // When the number of stop_source reaches 0, the remaining stop_tokens's
  // stop_possible will return false. We need this counter to track this.
  //
  // The "callback list locked" bit implements the atomic_unique_lock to
  // guard the operations on the callback list
  //
  //       31 - 1          |    0           |
  //  stop_source counter  | stop_requested |
  atomic<uint32_t> __state_ = 0;
  std::mutex __mutex_;

  // Reference count for stop_token + stop_callback + stop_source
  // When the counter reaches zero, the state is destroyed
  // It is used by __intrusive_shared_ptr, but it is stored here for better layout
  atomic<uint32_t> __ref_count_ = 0;

  using __state_t            = uint32_t;
  using __callback_list_lock = __nothrow_mutex_lock;
  using __callback_list      = __intrusive_list_view<__stop_callback_base>;

  __callback_list __callback_list_;
  __thread_id __requesting_thread_;

public:
  _LIBCPP_HIDE_FROM_ABI __stop_state() noexcept = default;

  _LIBCPP_HIDE_FROM_ABI void __increment_stop_source_counter() noexcept {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __state_.load(std::memory_order_relaxed) <= static_cast<__state_t>(~(1 << __stop_source_counter_shift)),
        "stop_source's counter reaches the maximum. Incrementing the counter will overflow");
    __state_.fetch_add(1 << __stop_source_counter_shift, std::memory_order_relaxed);
  }

  // We are not destroying the object after counter decrements to zero, nor do we have
  // operations depend on the ordering of decrementing the counter. relaxed is enough.
  _LIBCPP_HIDE_FROM_ABI void __decrement_stop_source_counter() noexcept {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __state_.load(std::memory_order_relaxed) >= static_cast<__state_t>(1 << __stop_source_counter_shift),
        "stop_source's counter is 0. Decrementing the counter will underflow");
    __state_.fetch_sub(1 << __stop_source_counter_shift, std::memory_order_relaxed);
  }

  _LIBCPP_HIDE_FROM_ABI bool __stop_requested() const noexcept {
    // acquire because [thread.stoptoken.intro] A call to request_stop that returns true
    // synchronizes with a call to stop_requested on an associated stop_token or stop_source
    // object that returns true.
    // request_stop's compare_exchange_weak has release which syncs with this acquire
    return (__state_.load(std::memory_order_acquire) & __stop_requested_bit) != 0;
  }

  _LIBCPP_HIDE_FROM_ABI bool __stop_possible_for_stop_token() const noexcept {
    // [stoptoken.mem] false if "a stop request was not made and there are no associated stop_source objects"
    // Todo: Can this be std::memory_order_relaxed as the standard does not say anything except not to introduce data
    // race?
    __state_t __curent_state = __state_.load(std::memory_order_acquire);
    return ((__curent_state & __stop_requested_bit) != 0) || ((__curent_state >> __stop_source_counter_shift) != 0);
  }

  _LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI bool __request_stop() noexcept {
    __callback_list_lock __cb_list_lock(__mutex_);
    auto __old = __state_.fetch_or(__stop_requested_bit, std::memory_order_release);
    if ((__old & __stop_requested_bit) == __stop_requested_bit) {
      return false;
    }
    __requesting_thread_ = this_thread::get_id();

    while (!__callback_list_.__empty()) {
      auto __cb = __callback_list_.__pop_front();

      // allow other callbacks to be removed while invoking the current callback
      __cb_list_lock.__unlock();

      bool __destroyed   = false;
      __cb->__destroyed_ = &__destroyed;

      __cb->__invoke();

      // __cb's invoke function could potentially delete itself. We need to check before accessing __cb's member
      if (!__destroyed) {
        // needs to set __destroyed_ pointer to nullptr, otherwise it points to a local variable
        // which is to be destroyed at the end of the loop
        __cb->__destroyed_ = nullptr;

        // [stopcallback.cons] If callback is concurrently executing on another thread, then the return
        // from the invocation of callback strongly happens before ([intro.races]) callback is destroyed.
        // this release syncs with the acquire in the remove_callback
        __cb->__completed_.store(true, std::memory_order_release);
        __cb->__completed_.notify_all();
      }

      __cb_list_lock.__lock();
    }

    return true;
  }

  _LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI bool __add_callback(__stop_callback_base* __cb) noexcept {
    __callback_list_lock __cb_list_lock(__mutex_);
    auto __state = __state_.load(std::memory_order_acquire);
    if ((__state & __stop_requested_bit) != 0) {
      // already stop requested, synchronously run the callback and no need to lock the list again
      __cb->__invoke();
      return false;
    }

    if ((__state >> __stop_source_counter_shift) == 0) {
      return false;
    }

    __callback_list_.__push_front(__cb);

    return true;
    // unlock here: [thread.stoptoken.intro] Registration of a callback synchronizes with the invocation of
    // that callback.
    // Note: this release sync with the acquire in the request_stop' __try_lock_for_request_stop
  }

  // called by the destructor of stop_callback
  _LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void __remove_callback(__stop_callback_base* __cb) noexcept {
    __callback_list_lock __cb_list_lock(__mutex_);

    // under below condition, the request_stop call just popped __cb from the list and could execute it now
    bool __potentially_executing_now = __cb->__prev_ == nullptr && !__callback_list_.__is_head(__cb);

    if (__potentially_executing_now) {
      auto __requested_thread = __requesting_thread_;
      __cb_list_lock.__unlock();

      if (std::this_thread::get_id() != __requested_thread) {
        // [stopcallback.cons] If callback is concurrently executing on another thread, then the return
        // from the invocation of callback strongly happens before ([intro.races]) callback is destroyed.
        __cb->__completed_.wait(false, std::memory_order_acquire);
      } else {
        // The destructor of stop_callback runs on the same thread of the thread that invokes the callback.
        // The callback is potentially invoking its own destuctor. Set the flag to avoid accessing destroyed
        // members on the invoking side
        if (__cb->__destroyed_) {
          *__cb->__destroyed_ = true;
        }
      }
    } else {
      __callback_list_.__remove(__cb);
    }
  }

  template <class _Tp>
  friend struct __intrusive_shared_ptr_traits;
};

template <class _Tp>
struct __intrusive_shared_ptr_traits;

template <>
struct __intrusive_shared_ptr_traits<__stop_state> {
  _LIBCPP_HIDE_FROM_ABI static atomic<uint32_t>& __get_atomic_ref_count(__stop_state& __state) {
    return __state.__ref_count_;
  }
};

#endif // _LIBCPP_STD_VER >= 20 && !defined(_LIBCPP_HAS_NO_THREADS)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___STOP_TOKEN_STOP_STATE_H
