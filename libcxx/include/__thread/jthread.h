// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___THREAD_JTHREAD_H
#define _LIBCPP___THREAD_JTHREAD_H

#include <__config>
#include <__stop_token/stop_source.h>
#include <__stop_token/stop_token.h>
#include <__thread/id.h>
#include <__thread/support.h>
#include <__thread/thread.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

class jthread {
public:
  // types
  using id                 = thread::id;
  using native_handle_type = thread::native_handle_type;

  // [thread.jthread.cons], constructors, move, and assignment
  jthread() noexcept : __stop_source_(std::nostopstate) {}

  template <class _Fun, class... _Args>
  explicit jthread(_Fun&& __fun, _Args&&... __args)
    requires(!std::is_same_v<remove_cvref_t<_Fun>, jthread>)
      : __stop_source_(),
        __thread_(__init_thread(__stop_source_, std::forward<_Fun>(__fun), std::forward<_Args>(__args)...)) {
    static_assert(is_constructible_v<decay_t<_Fun>, _Fun>);
    static_assert((is_constructible_v<decay_t<_Args>, _Args> && ...));
    static_assert(is_invocable_v<decay_t<_Fun>, decay_t<_Args>...> ||
                  is_invocable_v<decay_t<_Fun>, stop_token, decay_t<_Args>...>);
  }

  ~jthread() {
    if (joinable()) {
      request_stop();
      join();
    }
  }

  jthread(const jthread&) = delete;

  jthread(jthread&&) noexcept = default;

  jthread& operator=(const jthread&) = delete;

  jthread& operator=(jthread&& __other) noexcept {
    if (this != &__other) {
      if (joinable()) {
        request_stop();
        join();
      }
      __stop_source_ = std::move(__other.__stop_source_);
      __thread_      = std::move(__other.__thread_);
    }

    return *this;
  }

  // [thread.jthread.mem], members
  void swap(jthread& __other) noexcept {
    std::swap(__stop_source_, __other.__stop_source_);
    std::swap(__thread_, __other.__thread_);
  }

  [[nodiscard]] bool joinable() const noexcept { return get_id() != id(); }

  void join() { __thread_.join(); }

  void detach() { __thread_.detach(); }

  [[nodiscard]] id get_id() const noexcept { return __thread_.get_id(); }

  [[nodiscard]] native_handle_type native_handle() { return __thread_.native_handle(); }

  // [thread.jthread.stop], stop token handling
  [[nodiscard]] stop_source get_stop_source() noexcept { return __stop_source_; }

  [[nodiscard]] stop_token get_stop_token() const noexcept { return __stop_source_.get_token(); }

  bool request_stop() noexcept { return __stop_source_.request_stop(); }

  // [thread.jthread.special], specialized algorithms
  friend void swap(jthread& __lhs, jthread& __rhs) noexcept { __lhs.swap(__rhs); }

  // [thread.jthread.static], static members
  [[nodiscard]] static unsigned int hardware_concurrency() noexcept { return thread::hardware_concurrency(); }

private:
  template <class _Fun, class... _Args>
  static thread __init_thread(const stop_source& __ss, _Fun&& __fun, _Args&&... __args) {
    if constexpr (is_invocable_v<decay_t<_Fun>, stop_token, decay_t<_Args>...>) {
      return thread(std::forward<_Fun>(__fun), __ss.get_token(), std::forward<_Args>(__args)...);
    } else {
      return thread(std::forward<_Fun>(__fun), std::forward<_Args>(__args)...);
    }
  }

  stop_source __stop_source_;
  thread __thread_;
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS

_LIBCPP_POP_MACROS

#endif // _LIBCPP___THREAD_JTHREAD_H
