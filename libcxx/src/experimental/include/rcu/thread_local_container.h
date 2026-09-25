// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RCU_THREAD_LOCAL_CONTAINER_H
#define _LIBCPP___RCU_THREAD_LOCAL_CONTAINER_H

#include <__config>
#include <__functional/function_ref.h>
#include <__rcu/rcu_domain.h>

#include <mutex>
#include <optional>
#include <vector>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

// Tp must be thread-safe itself between
// - the operation that is done by the object from get_current_thread_instance calls
// - and the operation that for_each
// since there is no mutex guarding between them
template <class Tp>
class thread_local_container {
public:
  struct thread_entry {
    Tp instance_;
    thread_local_container& container_;
    function_ref<void(Tp&) noexcept> pre_dtor_callback_;

    thread_entry(thread_local_container& container, function_ref<void(Tp&) noexcept> cb)
        : instance_(), container_(container), pre_dtor_callback_(cb) {
      container_.register_instance(instance_);
    }
    thread_entry(const thread_entry&) = delete;
    thread_entry(thread_entry&&)      = delete;

    ~thread_entry() {
      pre_dtor_callback_(instance_);
      container_.deregister_instance(instance_);
    }
  };

  using thread_local_t = optional<thread_entry>;
  using get_thread_entry_fn = thread_local_t&();

private:
  // Keep track of all thread-local instances
  // Only emplaced the first time a thread is trying to access its thread-local instance.
  get_thread_entry_fn* get_thread_entry_;
  vector<Tp*> instances_{};
  mutable mutex mtx_{};

  void register_instance(Tp& obj) {
    lock_guard<std::mutex> lg(mtx_);
    instances_.emplace_back(&obj);
  }

  void deregister_instance(Tp& obj) {
    lock_guard<std::mutex> lg(mtx_);
    instances_.erase(
        std::remove_if(instances_.begin(), instances_.end(), [&obj](Tp* instance) { return instance == &obj; }),
        instances_.end());
  }

  static void empty_callback(Tp&) noexcept {}

public:
  thread_local_container(get_thread_entry_fn* get_thread_entry) : get_thread_entry_(get_thread_entry) {}
  thread_local_container(thread_local_container&&) = delete;

  Tp& get_current_thread_instance(function_ref<void(Tp&) noexcept> cb = cw<&empty_callback>) {
    auto& thread_entry = get_thread_entry_();
    if (!thread_entry.has_value()) {
      auto& entry = thread_entry.emplace(*this, cb);
      return entry.instance_;
    }
    return thread_entry->instance_;
  }

  template <class Func>
  void for_each(Func&& f) {
    unique_lock<std::mutex> lock(mtx_);
    for (auto instance : instances_) {
      f(*instance);
    }
  }
};

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_THREAD_LOCAL_CONTAINER_H
