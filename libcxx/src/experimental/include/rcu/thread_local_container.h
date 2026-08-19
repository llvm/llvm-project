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
#include <__functional/function.h>
#include <__rcu/rcu_domain.h>

#include <atomic>
#include <optional>
#include <mutex>
#include <vector>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

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

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_EXPERIMENTAL_RCU

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RCU_THREAD_LOCAL_CONTAINER_H
