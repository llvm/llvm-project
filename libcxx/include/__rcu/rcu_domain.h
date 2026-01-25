// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RCU_RCU_DOMAIN_H
#define _LIBCPP___RCU_RCU_DOMAIN_H

#include <__config>

#include <__atomic/atomic.h>
#include <__rcu/rcu_list.h>

// todo replace with internal headers
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

// todo debug
#include <cstdio>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

template <class _Tp>
class __thread_local_owner {
  // todo put globals in experimental dylib
  inline static thread_local map<const __thread_local_owner*, _Tp> __thread_local_instances;

  // Keep track of all thread-local instances owned by this owner.
  // Only emplaced the first time a thread is trying to access its thread-local instance.
  vector<atomic_ref<_Tp>> __owned_instances_;
  mutex __mtx_;

  void __register(_Tp& __obj) {
    lock_guard<std::mutex> __lg(__mtx_);
    __owned_instances_.emplace_back(__obj);
  }

  // todo: deregister on thread exit?

public:
  __thread_local_owner()                       = default;
  __thread_local_owner(__thread_local_owner&&) = delete;

  atomic_ref<_Tp> __get_current_thread_instance() {
    auto __it = __thread_local_instances.find(this);
    if (__it == __thread_local_instances.end()) {
      auto [new_it, _] = __thread_local_instances.try_emplace(this, _Tp());
      auto& __obj      = new_it->second;
      __register(__obj);
      return atomic_ref(__obj);
    }
    return atomic_ref(__it->second);
  }

  template <class _Func>
  void __for_each_owned_instances(_Func&& __f) {
    unique_lock<std::mutex> __lock(__mtx_);
    for (auto __instance : __owned_instances_) {
      __f(__instance);
    }
  }
};

// Adopted the 2-phase implementation in the section
// "3) General-Purpose RCU" of the paper
// http://www.rdrop.com/users/paulmck/RCU/urcu-supp-accepted.2011.08.30a.pdf

struct __reader_states {
  // bit 15 is the grace period phase 0 or 1
  // bits 0-14 is the reader nest level
  //
  // a thread can have nested reader locks, such as
  // domain.lock();   // nest level = 1
  // domain.lock();   // nest level = 2
  // ...
  // domain.unlock(); // nest level = 1
  // domain.unlock(); // nest level = 0

  static constexpr uint16_t __grace_period_phase_mask = 0b1000'0000'0000'0000;
  static constexpr uint16_t __reader_nest_level_mask  = 0b0111'1111'1111'1111;

  using __state_type = uint16_t;

  __thread_local_owner<__state_type> __per_thread_states_;

  static uint16_t __get_grace_period_phase(__state_type __state) { return __state & __grace_period_phase_mask; }

  static uint16_t __get_reader_nest_level(__state_type __state) { return __state & __reader_nest_level_mask; }

  static bool __is_quiescent_state(__state_type __state) { return __get_reader_nest_level(__state) == 0; }
};

class rcu_domain {
  // todo optimize the layout

  __reader_states __reader_states_;

  // only the highest bit is used for the phase.
  std::atomic<__reader_states::__state_type> __global_reader_phase_{};

  // only one thread is allowed to update concurrently
  std::mutex __grace_period_mutex_; // todo this is not noexcept

  std::atomic<bool> __grace_period_waiting_flag_ = false;

  // todo: maybe use a lock-free queue
  std::mutex __retire_queue_mutex_; // todo this is not noexcept
  __rcu_singly_list_view __retired_callback_queue_;

  // these two queues do not need extra synchronization
  // as they are always processed under the grace period mutex
  __rcu_singly_list_view __callbacks_phase_1_;
  __rcu_singly_list_view __callbacks_phase_2_;

  rcu_domain() = default;

  friend struct __rcu_domain_access;
  template <class, class>
  friend class rcu_obj_base;

  // todo put globals in dylib
  static rcu_domain& __rcu_default_domain() noexcept {
    static rcu_domain __default_domain;
    return __default_domain;
  }

  template <class Callback>
  void __retire_callback(Callback&& __cb) noexcept {
    auto* __node        = new __rcu_node();
    __node->__callback_ = std::forward<Callback>(__cb);
    std::unique_lock __lk(__retire_queue_mutex_);
    __retired_callback_queue_.__push_back(__node);
  }

  void __synchronize() noexcept {
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    std::unique_lock __lk(__grace_period_mutex_);
    //std::printf("rcu_domain::__synchronize() going through phase 1\n");
    auto __ready_callbacks = __update_phase_and_wait();

    // Invoke the ready callbacks outside of the grace period mutex
    __lk.unlock();
    __ready_callbacks.__for_each([](auto* __node) { __node->__callback_(); });
    __lk.lock();

    __barrier();
    //std::printf("rcu_domain::__synchronize() going through phase 2\n");
    __ready_callbacks = __update_phase_and_wait();

    // Invoke the ready callbacks outside of the grace period mutex
    __lk.unlock();
    __ready_callbacks.__for_each([](auto* __node) { __node->__callback_(); });
    __cxx_atomic_thread_fence(memory_order_seq_cst);
  }

  __rcu_singly_list_view __update_phase_and_wait() noexcept {
    std::unique_lock __retire_lk(__retire_queue_mutex_);
    __callbacks_phase_1_.__splice_back(__retired_callback_queue_);
    __retire_lk.unlock();

    // Flip the global phase
    auto __old_phase =
        __global_reader_phase_.fetch_xor(__reader_states::__grace_period_phase_mask, std::memory_order_relaxed);
    auto __new_phase = __old_phase ^ __reader_states::__grace_period_phase_mask;
    //std::printf("rcu_domain::__update_phase_and_wait() new phase: 0x%04x\n", __new_phase);

    __barrier();
    // Wait for all threads to quiesce in the old phase
    while (__any_reader_in_ongoing_grace_period(__new_phase)) {
      __grace_period_waiting_flag_.store(true, std::memory_order_relaxed);
      __grace_period_waiting_flag_.wait(true, std::memory_order_relaxed);
    }
    __grace_period_waiting_flag_.store(false, std::memory_order_relaxed);

    __rcu_singly_list_view __ready_callbacks;
    __ready_callbacks.__splice_back(__callbacks_phase_2_);
    __callbacks_phase_2_.__splice_back(__callbacks_phase_1_);
    return __ready_callbacks;
  }

  bool __any_reader_in_ongoing_grace_period(__reader_states::__state_type __global_phase) noexcept {
    bool __any_ongoing = false;
    __reader_states_.__per_thread_states_.__for_each_owned_instances(
        [this, __global_phase, &__any_ongoing](atomic_ref<__reader_states::__state_type> __state) {
          if (__is_grace_period_ongoing(__state.load(memory_order_relaxed), __global_phase)) {
            __any_ongoing = true;
          }
        });
    return __any_ongoing;
  }

  bool __is_grace_period_ongoing(__reader_states::__state_type __thread_state,
                                 __reader_states::__state_type __global_phase) const noexcept {
    // https://lwn.net/Articles/323929/
    // The phase is flipped at the beginning of a grace period.
    // Any readers that started before the flip will have the old phase
    // and we consider them as ongoing that we need to wait for before we can close the grace period.
    return !__reader_states::__is_quiescent_state(__thread_state) &&
           __reader_states::__get_grace_period_phase(__thread_state) != __global_phase;
  }

  void __barrier() noexcept { asm volatile("" : : : "memory"); }

public:
  rcu_domain(const rcu_domain&)            = delete;
  rcu_domain& operator=(const rcu_domain&) = delete;

  void printAllReaderStatesInHex() {
    __reader_states_.__per_thread_states_.__for_each_owned_instances([](auto __state_ref) {
      std::printf("Reader state: 0x%04x\n", __state_ref.load());
    });
  }

  void lock() {
    auto __current_thread_state_ref = __reader_states_.__per_thread_states_.__get_current_thread_instance();

    if ((__reader_states::__is_quiescent_state(__current_thread_state_ref.load(memory_order_relaxed)))) {
      // Entering a read-side critical section from a quiescent state.
      __current_thread_state_ref.store(
          __reader_states::__state_type(__global_reader_phase_.load(std::memory_order_relaxed) | uint16_t(1)),
          memory_order_relaxed);
      __cxx_atomic_thread_fence(memory_order_seq_cst);
    } else {
      // Already in read-side critical section, just increment the nest level.
      __current_thread_state_ref.fetch_add(1, memory_order_relaxed);
    }
  }

  bool try_lock() {
    lock();
    return true;
  }

  void unlock() {
    auto __current_thread_state_ref = __reader_states_.__per_thread_states_.__get_current_thread_instance();
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    // Decrement the nest level.
    auto __old_state = __current_thread_state_ref.fetch_sub(1, memory_order_relaxed);

    if (__reader_states::__get_reader_nest_level(__old_state) == 1 &&
        __grace_period_waiting_flag_.load(memory_order_relaxed)) {
      // Transitioning to quiescent state, wake up waiters.
      __grace_period_waiting_flag_.store(false, std::memory_order_relaxed);
      __grace_period_waiting_flag_.notify_all();
    }
  }
};

struct __rcu_domain_access {
  static rcu_domain& __rcu_default_domain() noexcept { return rcu_domain::__rcu_default_domain(); }
  static void __rcu_synchronize(rcu_domain& __dom) noexcept { __dom.__synchronize(); }
};

// todo put it in the experimental dylib
inline rcu_domain& rcu_default_domain() noexcept { return __rcu_domain_access::__rcu_default_domain(); }

// todo put it in the experimental dylib
inline void rcu_synchronize(rcu_domain& __dom = rcu_default_domain()) noexcept {
  __rcu_domain_access::__rcu_synchronize(__dom);
}

void rcu_barrier(rcu_domain& dom = rcu_default_domain()) noexcept;

template <class T, class D = default_delete<T>>
void rcu_retire(T* p, D d = D(), rcu_domain& dom = rcu_default_domain());

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_RCU_DOMAIN_H
