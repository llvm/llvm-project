//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <rcu>
#include <vector>

#include "include/rcu/rcu_list.h"

_LIBCPP_BEGIN_NAMESPACE_STD

namespace {

template <class Tp>
class thread_local_container {
  struct thread_entry {
    Tp instance_;

    thread_entry() : instance_() { register_instance(instance_); }
    thread_entry(const thread_entry&) = delete;
    thread_entry(thread_entry&&)      = delete;

    ~thread_entry() { deregister_instance(instance_); }
  };

  inline static thread_local optional<thread_entry> thread_entry_{};

  // Keep track of all thread-local instances
  // Only emplaced the first time a thread is trying to access its thread-local instance.
  inline static vector<Tp*> instances_;
  inline static mutex mtx_;

  static void register_instance(Tp& obj) {
    lock_guard<std::mutex> lg(mtx_);
    instances_.emplace_back(&obj);
  }

  static void deregister_instance(Tp& obj) {
    lock_guard<std::mutex> lg(mtx_);
    instances_.erase(remove_if(instances_.begin(), instances_.end(), [&obj](Tp* instance) { return instance == &obj; }),
                     instances_.end());
  }

public:
  thread_local_container()                         = delete;
  thread_local_container(thread_local_container&&) = delete;

  static atomic_ref<Tp> get_current_thread_instance() {
    if (!thread_entry_.has_value()) {
      auto& entry = thread_entry_.emplace();
      return atomic_ref(entry.instance_);
    }
    return atomic_ref(thread_entry_->instance_);
  }

  template <class Func>
  static void for_each(Func&& f) {
    unique_lock<std::mutex> lock(mtx_);
    for (auto instance : instances_) {
      f(atomic_ref(*instance));
    }
  }
};

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

  static uint16_t get_grace_period_phase(state_type state) { return state & grace_period_phase_mask; }

  static uint16_t get_reader_nest_level(state_type state) { return state & reader_nest_level_mask; }

  static bool is_quiescent_state(state_type state) { return get_reader_nest_level(state) == 0; }

  static state_type make_state(state_type grace_period_phase, state_type reader_nest_level) {
    return (grace_period_phase & grace_period_phase_mask) | (reader_nest_level & reader_nest_level_mask);
  }
};

class rcu_domain_impl {
  using per_thread_states = thread_local_container<reader_states::state_type>;

  // only the highest bit is used for the phase.
  std::atomic<reader_states::state_type> global_reader_phase_{};

  // only one writer thread is allowed to call synchronize concurrently
  std::mutex grace_period_mutex_; // todo this is not noexcept

  // flag used for waking up writer threads waiting for all reader threads' quiescent state
  std::atomic<bool> grace_period_waiting_flag_ = false;

  // todo: maybe use a lock-free queue
  std::mutex retire_queue_mutex_; // todo this is not noexcept
  rcu_singly_list_view __retired_callback_queue_;

  // these two queues do not need extra synchronization
  // as they are always processed under the grace period mutex
  rcu_singly_list_view callbacks_phase_1_;
  rcu_singly_list_view callbacks_phase_2_;

  friend class rcu_domain;

  rcu_singly_list_view update_phase_and_wait() noexcept {
    std::unique_lock retire_lk(retire_queue_mutex_);
    callbacks_phase_1_.__splice_back(__retired_callback_queue_);
    retire_lk.unlock();

    // Flip the global phase
    auto old_phase = global_reader_phase_.fetch_xor(reader_states::grace_period_phase_mask, std::memory_order_relaxed);
    auto __new_phase = old_phase ^ reader_states::grace_period_phase_mask;
    // std::printf("rcu_domain::update_phase_and_wait() new phase: 0x%04x\n", __new_phase);

    __barrier();
    // Wait for all threads to quiesce in the old phase
    while (any_reader_in_ongoing_grace_period(__new_phase)) {
      grace_period_waiting_flag_.store(true, std::memory_order_relaxed);
      grace_period_waiting_flag_.wait(true, std::memory_order_relaxed);
    }
    grace_period_waiting_flag_.store(false, std::memory_order_relaxed);

    rcu_singly_list_view ready_callbacks;
    ready_callbacks.__splice_back(callbacks_phase_2_);
    callbacks_phase_2_.__splice_back(callbacks_phase_1_);
    return ready_callbacks;
  }

  bool any_reader_in_ongoing_grace_period(reader_states::state_type __global_phase) noexcept {
    bool any_ongoing = false;
    per_thread_states::for_each([this, __global_phase, &any_ongoing](atomic_ref<reader_states::state_type> __state) {
      if (is_grace_period_ongoing(__state.load(memory_order_relaxed), __global_phase)) {
        any_ongoing = true;
      }
    });
    return any_ongoing;
  }

  bool is_grace_period_ongoing(reader_states::state_type thread_state,
                               reader_states::state_type global_phase) const noexcept {
    // https://lwn.net/Articles/323929/
    // The phase is flipped at the beginning of a grace period.
    // Any readers that started before the flip will have the old phase
    // and we consider them as ongoing that we need to wait for before we can close the grace period.
    return !reader_states::is_quiescent_state(thread_state) &&
           reader_states::get_grace_period_phase(thread_state) != global_phase;
  }

  void __barrier() noexcept { asm volatile("" : : : "memory"); }

public:
  void lock() noexcept {
    auto current_thread_state_ref = per_thread_states::get_current_thread_instance();

    if ((reader_states::is_quiescent_state(current_thread_state_ref.load(memory_order_relaxed)))) {
      // Entering a read-side critical section from a quiescent state.
      current_thread_state_ref.store(
          reader_states::make_state(global_reader_phase_.load(std::memory_order_relaxed), 1), memory_order_relaxed);
      __cxx_atomic_thread_fence(memory_order_seq_cst);
    } else {
      // Already in read-side critical section, just increment the nest level.
      current_thread_state_ref.fetch_add(1, memory_order_relaxed);
    }
  }

  void unlock() noexcept {
    auto current_thread_state_ref = per_thread_states::get_current_thread_instance();
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    // Decrement the nest level.
    auto old_state = current_thread_state_ref.fetch_sub(1, memory_order_relaxed);

    if (reader_states::get_reader_nest_level(old_state) == 1 && grace_period_waiting_flag_.load(memory_order_relaxed)) {
      // Transitioning to quiescent state, wake up waiters.
      grace_period_waiting_flag_.store(false, std::memory_order_relaxed);
      grace_period_waiting_flag_.notify_all();
    }
  }

  void retire(__rcu_node* node) noexcept {
    lock_guard<std::mutex> lk(retire_queue_mutex_);
    __retired_callback_queue_.__push_back(node);
  }

  void synchronize() noexcept {
    __cxx_atomic_thread_fence(memory_order_seq_cst);
    std::unique_lock lk(grace_period_mutex_);

    auto ready_callbacks = update_phase_and_wait();

    // Invoke the ready callbacks outside of the grace period mutex
    lk.unlock();
    ready_callbacks.__for_each([](auto* node) { node->__callback_(); });
    lk.lock();

    __barrier();
    ready_callbacks = update_phase_and_wait();

    // Invoke the ready callbacks outside of the grace period mutex
    lk.unlock();
    ready_callbacks.__for_each([](auto* node) { node->__callback_(); });
    __cxx_atomic_thread_fence(memory_order_seq_cst);
  }

  void printAllReaderStatesInHex() {
    per_thread_states::for_each([](auto __state_ref) { std::printf("Reader state: 0x%04x\n", __state_ref.load()); });
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

void rcu_domain::printAllReaderStatesInHex() { __pimpl_->printAllReaderStatesInHex(); }

void rcu_domain::lock() noexcept { __pimpl_->lock(); }

void rcu_domain::unlock() noexcept { __pimpl_->unlock(); }

void rcu_domain::__retire(__rcu_node* node) noexcept { __pimpl_->retire(node); }

rcu_domain& rcu_default_domain() noexcept { return rcu_domain::__rcu_default_domain(); }

void rcu_synchronize(rcu_domain& dom) noexcept { dom.__pimpl_->synchronize(); }

void rcu_barrier(rcu_domain& dom) noexcept { rcu_synchronize(dom); }

_LIBCPP_END_NAMESPACE_STD
