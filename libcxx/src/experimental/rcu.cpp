//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <rcu>

#include "include/rcu/rcu_list.h"
#include "include/rcu/thread_local_container.h"

// todo: remove debug print
#include <cstdio>
//

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

namespace {

// Adopted the 2-phase implementation in the section
// "3) General-Purpose RCU" of the paper
// http://www.rdrop.com/users/paulmck/RCU/urcu-supp-accepted.2011.08.30a.pdf

struct reader_states {
  // bit 15 is the grace period phase 0 or 1
  // bits 0-14 is the reader nest level
  //
  // a thread can have nested reader locks, such as
  // domain.lock();   // nest level = 1
  // domain.lock();   // nest level = 2
  // ...
  // domain.unlock(); // nest level = 1
  // domain.unlock(); // nest level = 0

  static constexpr uint16_t grace_period_phase_mask = 0b1000'0000'0000'0000;
  static constexpr uint16_t reader_nest_level_mask  = 0b0111'1111'1111'1111;

  using state_type = uint16_t;

  std::atomic<state_type> state_{};

  bool is_quiescent_state() const noexcept {
    return get_reader_nest_level(state_.load(std::memory_order_relaxed)) == 0;
  }

  bool is_grace_period_ongoing_at_phase(state_type global_phase) const noexcept {
    auto state = state_.load(std::memory_order_relaxed);
    return !is_quiescent_state(state) && get_grace_period_phase(state) != global_phase;
  }

  void set_state(state_type grace_period_phase, state_type reader_nest_level) {
    auto state = (grace_period_phase & grace_period_phase_mask) | (reader_nest_level & reader_nest_level_mask);
    state_.store(state, memory_order_relaxed);
  }

  void increment_nest_level() noexcept { state_.fetch_add(1, memory_order_relaxed); }

  // return previous nested level
  uint16_t decrement_nest_level() noexcept {
    auto old_state = state_.fetch_sub(1, memory_order_relaxed);
    return get_reader_nest_level(old_state);
  }

  state_type debug_get_state() const noexcept { return state_.load(); }

private:
  static uint16_t get_grace_period_phase(state_type state) noexcept { return state & grace_period_phase_mask; }
  static bool is_quiescent_state(state_type state) noexcept { return get_reader_nest_level(state) == 0; }

  static uint16_t get_reader_nest_level(state_type state) noexcept { return state & reader_nest_level_mask; }
};

class rcu_domain_impl {
  using per_thread_states = thread_local_container<reader_states>;

  // only the highest bit is used for the phase.
  std::atomic<reader_states::state_type> global_reader_phase_{};

  // only one writer thread is allowed to call synchronize concurrently
  std::mutex grace_period_mutex_; // todo this is not noexcept

  // flag used for waking up writer threads waiting for all reader threads' quiescent state
  std::atomic<bool> grace_period_waiting_flag_ = false;

  using per_thread_retired_queue_stage0 = thread_local_container<rcu_atomic_list_view>;

  // these two queues do not need extra synchronization
  // as they are always processed under the grace period mutex
  rcu_singly_list_view retired_queue_stage1_;
  rcu_singly_list_view retired_queue_stage2_;

  friend class rcu_domain;

  void update_phase_and_wait() noexcept {
    rcu_singly_list_view working_queue;
    per_thread_retired_queue_stage0::for_each([&working_queue](rcu_atomic_list_view& stage0_list) {
      working_queue.splice_back(stage0_list);
    });

    // Flip the global phase
    auto old_phase = global_reader_phase_.fetch_xor(reader_states::grace_period_phase_mask, std::memory_order_relaxed);
    auto new_phase = old_phase ^ reader_states::grace_period_phase_mask;
    // std::printf("rcu_domain::update_phase_and_wait() new phase: 0x%04x\n", new_phase);

    std::atomic_signal_fence(std::memory_order_seq_cst);

    // Wait for all threads to quiesce in the old phase
    while (any_reader_in_ongoing_grace_period(new_phase)) {
      grace_period_waiting_flag_.store(true, std::memory_order_relaxed);
      grace_period_waiting_flag_.wait(true, std::memory_order_relaxed);
    }
    grace_period_waiting_flag_.store(false, std::memory_order_relaxed);

    retired_queue_stage2_.splice_back(retired_queue_stage1_);
    retired_queue_stage1_.splice_back(working_queue);
  }

  bool any_reader_in_ongoing_grace_period(reader_states::state_type global_phase) noexcept {
    bool any_ongoing = false;
    per_thread_states::for_each([global_phase, &any_ongoing](reader_states& state) {
      if (state.is_grace_period_ongoing_at_phase(global_phase)) {
        any_ongoing = true;
      }
    });
    return any_ongoing;
  }

public:
  void lock() noexcept {
    reader_states& current_thread_state = per_thread_states::get_current_thread_instance();

    if (current_thread_state.is_quiescent_state()) {
      // Enter critical section by setting the nested level from 0 -> 1
      current_thread_state.set_state(global_reader_phase_.load(memory_order_relaxed), 1);
      std::atomic_thread_fence(memory_order_seq_cst);
    } else {
      // Already in read-side critical section, just increment the nest level.
      current_thread_state.increment_nest_level();
    }
  }

  void unlock() noexcept {
    reader_states& current_thread_state = per_thread_states::get_current_thread_instance();
    std::atomic_thread_fence(memory_order_seq_cst);
    // Decrement the nest level.
    auto old_nested_level = current_thread_state.decrement_nest_level();

    if (old_nested_level == 1 && grace_period_waiting_flag_.load(memory_order_relaxed)) {
      // Transitioning to quiescent state, wake up waiters.
      grace_period_waiting_flag_.store(false, std::memory_order_relaxed);
      grace_period_waiting_flag_.notify_all();
    }
  }

  void retire(__rcu_node* node) noexcept {
    rcu_atomic_list_view& stage0_queue = per_thread_retired_queue_stage0::get_current_thread_instance();
    stage0_queue.push_front(node);
  }

  void synchronize(bool invoke_callback) noexcept {
    std::atomic_thread_fence(memory_order_seq_cst);
    std::unique_lock lk(grace_period_mutex_);

    update_phase_and_wait();

    if (invoke_callback) {
      rcu_singly_list_view ready_callbacks;
      ready_callbacks.splice_back(retired_queue_stage2_);
      ready_callbacks.for_each([](auto* node) { node->__callback_(node); });
    }

    std::atomic_signal_fence(memory_order_seq_cst);
    update_phase_and_wait();

    if (invoke_callback) {
      rcu_singly_list_view ready_callbacks;
      ready_callbacks.splice_back(retired_queue_stage2_);
      ready_callbacks.for_each([](auto* node) { node->__callback_(node); });
    }
    std::atomic_thread_fence(memory_order_seq_cst);
  }

  void __debug_print_all_reader_states_in_hex() {
    per_thread_states::for_each([](reader_states& state) {
      std::printf("Reader state: 0x%04x\n", state.debug_get_state());
    });
  }
};
} // namespace

class rcu_domain::__impl : public rcu_domain_impl {};

rcu_domain& rcu_domain::__rcu_default_domain() noexcept {
  static rcu_domain default_domain;
  return default_domain;
}

rcu_domain::rcu_domain() : __pimpl_(std::make_unique<__impl>()) {}
rcu_domain::~rcu_domain() = default;

void rcu_domain::__debug_print_all_reader_states_in_hex() { __pimpl_->__debug_print_all_reader_states_in_hex(); }

void rcu_domain::lock() noexcept { __pimpl_->lock(); }

void rcu_domain::unlock() noexcept { __pimpl_->unlock(); }

void rcu_domain::__retire(__rcu_node* node) noexcept { __pimpl_->retire(node); }

rcu_domain& rcu_default_domain() noexcept { return rcu_domain::__rcu_default_domain(); }

void rcu_synchronize(rcu_domain& dom) noexcept { dom.__pimpl_->synchronize(false); }

void rcu_barrier(rcu_domain& dom) noexcept { dom.__pimpl_->synchronize(true); }

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
