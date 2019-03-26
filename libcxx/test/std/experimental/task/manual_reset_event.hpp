// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_TEST_EXPERIMENTAL_TASK_MANUAL_RESET_EVENT
#define _LIBCPP_TEST_EXPERIMENTAL_TASK_MANUAL_RESET_EVENT

#include <experimental/coroutine>
#include <atomic>

// manual_reset_event is a coroutine synchronisation tool that allows one
// coroutine to await the event object and if the event was not crrently
// in the 'set' state then will suspend the awaiting coroutine until some
// thread calls .set() on the event.
class manual_reset_event
{
  friend class _Awaiter;

  class _Awaiter
  {
  public:

    _Awaiter(const manual_reset_event* __event) noexcept
    : __event_(__event)
    {}

    bool await_ready() const noexcept
    {
      return __event_->is_set();
    }

    bool await_suspend(std::experimental::coroutine_handle<> __coro) noexcept
    {
      _LIBCPP_ASSERT(
        __event_->__state_.load(std::memory_order_relaxed) !=
        __State::__not_set_waiting_coroutine,
        "This manual_reset_event already has another coroutine awaiting it. "
        "Only one awaiting coroutine is supported."
      );

      __event_->__awaitingCoroutine_ = __coro;

      // If the compare-exchange fails then this means that the event was
      // already 'set' and so we should not suspend - this code path requires
      // 'acquire' semantics so we have visibility of writes prior to the
      // .set() operation that transitioned the event to the 'set' state.
      // If the compare-exchange succeeds then this needs 'release' semantics
      // so that a subsequent call to .set() has visibility of our writes
      // to the coroutine frame and to __event_->__awaitingCoroutine_ after
      // reading our write to __event_->__state_.
      _State oldState = _State::__not_set;
      return __event_->__state_.compare_exchange_strong(
        oldState,
        _State::__not_set_waiting_coroutine,
        std::memory_order_release,
        std::memory_order_acquire);
    }

    void await_resume() const noexcept {}

  private:
    const manual_reset_event* __event_;
  };

public:

  manual_reset_event(bool __initiallySet = false) noexcept
  : __state_(__initiallySet ? _State::__set : _State::__not_set)
  {}

  bool is_set() const noexcept
  {
    return __state_.load(std::memory_order_acquire) == _State::__set;
  }

  void set() noexcept
  {
    // Needs to be 'acquire' in case the old value was a waiting coroutine
    // so that we have visibility of the writes to the coroutine frame in
    // the current thrad before we resume it.
    // Also needs to be 'release' in case the old value was 'not-set' so that
    // another thread that subsequently awaits the
    _State oldState = __state_.exchange(_State::__set, std::memory_order_acq_rel);
    if (oldState == _State::__not_set_waiting_coroutine)
    {
      _VSTD::exchange(__awaitingCoroutine_, {}).resume();
    }
  }

  void reset() noexcept
  {
    _LIBCPP_ASSERT(
      __state_.load(std::memory_order_relaxed) != _State::__not_set_waiting_coroutine,
      "Illegal to call reset() if a coroutine is currently awaiting the event.");

    // Note, we use 'relaxed' memory order here since it considered a
    // data-race to call reset() concurrently either with operator co_await()
    // or with set().
    __state_.store(_State::__not_set, std::memory_order_relaxed);
  }

  auto operator co_await() const noexcept
  {
    return _Awaiter{ this };
  }

private:

  enum class _State {
    __not_set,
    __not_set_waiting_coroutine,
    __set
  };

  // TODO: Can we combine these two members into a single std::atomic<void*>?
  mutable std::atomic<_State> __state_;
  mutable std::experimental::coroutine_handle<> __awaitingCoroutine_;

};

#endif
