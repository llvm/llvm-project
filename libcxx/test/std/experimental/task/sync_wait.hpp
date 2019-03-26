// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_TEST_EXPERIMENTAL_TASK_SYNC_WAIT
#define _LIBCPP_TEST_EXPERIMENTAL_TASK_SYNC_WAIT

#include <experimental/__config>
#include <experimental/coroutine>
#include <type_traits>
#include <mutex>
#include <condition_variable>

#include "awaitable_traits.hpp"

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL_COROUTINES

// Thread-synchronisation helper that allows one thread to block in a call
// to .wait() until another thread signals the thread by calling .set().
class __oneshot_event
{
public:
  __oneshot_event() : __isSet_(false) {}

  void set() noexcept
  {
    unique_lock<mutex> __lock{ __mutex_ };
    __isSet_ = true;
    __cv_.notify_all();
  }

  void wait() noexcept
  {
    unique_lock<mutex> __lock{ __mutex_ };
    __cv_.wait(__lock, [this] { return __isSet_; });
  }

private:
  mutex __mutex_;
  condition_variable __cv_;
  bool __isSet_;
};

template<typename _Derived>
class __sync_wait_promise_base
{
public:

  using __handle_t = coroutine_handle<_Derived>;

private:

  struct _FinalAwaiter
  {
      bool await_ready() noexcept { return false; }
      void await_suspend(__handle_t __coro) noexcept
      {
        __sync_wait_promise_base& __promise = __coro.promise();
        __promise.__event_.set();
      }
      void await_resume() noexcept {}
  };

  friend struct _FinalAwaiter;

public:

  __handle_t get_return_object() { return __handle(); }
  suspend_always initial_suspend() { return {}; }
  _FinalAwaiter final_suspend() { return {}; }

private:

  __handle_t __handle() noexcept
  {
    return __handle_t::from_promise(static_cast<_Derived&>(*this));
  }

protected:

  // Start the coroutine and then block waiting for it to finish.
  void run() noexcept
  {
    __handle().resume();
    __event_.wait();
  }

private:

  __oneshot_event __event_;

};

template<typename _Tp>
class __sync_wait_promise final
  : public __sync_wait_promise_base<__sync_wait_promise<_Tp>>
{
public:

  __sync_wait_promise() : __state_(_State::__empty) {}

  ~__sync_wait_promise()
  {
    switch (__state_)
    {
      case _State::__empty:
      case _State::__value:
        break;
#ifndef _LIBCPP_NO_EXCEPTIONS
      case _State::__exception:
        __exception_.~exception_ptr();
        break;
#endif
    }
  }

  void return_void() noexcept
  {
    // Should be unreachable since coroutine should always
    // suspend at `co_yield` point where it will be destroyed
    // or will fail with an exception and bypass return_void()
    // and call unhandled_exception() instead.
    std::abort();
  }

  void unhandled_exception() noexcept
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
    ::new (static_cast<void*>(&__exception_)) exception_ptr(
      std::current_exception());
    __state_ = _State::__exception;
#else
    _VSTD::abort();
#endif
  }

  auto yield_value(_Tp&& __value) noexcept
  {
    __valuePtr_ = std::addressof(__value);
    __state_ = _State::__value;
    return this->final_suspend();
  }

  _Tp&& get()
  {
    this->run();

#ifndef _LIBCPP_NO_EXCEPTIONS
    if (__state_ == _State::__exception)
    {
      std::rethrow_exception(_VSTD::move(__exception_));
    }
#endif

    return static_cast<_Tp&&>(*__valuePtr_);
  }

private:

  enum class _State {
    __empty,
    __value,
    __exception
  };

  _State __state_;
  union {
    std::add_pointer_t<_Tp> __valuePtr_;
    std::exception_ptr __exception_;
  };

};

template<>
struct __sync_wait_promise<void> final
  : public __sync_wait_promise_base<__sync_wait_promise<void>>
{
public:

  void unhandled_exception() noexcept
  {
#ifndef _LIBCPP_NO_EXCEPTIONS
    __exception_ = std::current_exception();
#endif
  }

  void return_void() noexcept {}

  void get()
  {
    this->run();

#ifndef _LIBCPP_NO_EXCEPTIONS
    if (__exception_)
    {
      std::rethrow_exception(_VSTD::move(__exception_));
    }
#endif
  }

private:

  std::exception_ptr __exception_;

};

template<typename _Tp>
class __sync_wait_task final
{
public:
  using promise_type = __sync_wait_promise<_Tp>;

private:
  using __handle_t = typename promise_type::__handle_t;

public:

  __sync_wait_task(__handle_t __coro) noexcept : __coro_(__coro) {}

  ~__sync_wait_task()
  {
    _LIBCPP_ASSERT(__coro_, "Should always have a valid coroutine handle");
    __coro_.destroy();
  }

  decltype(auto) get()
  {
    return __coro_.promise().get();
  }
private:
  __handle_t __coro_;
};

template<typename _Tp>
struct __remove_rvalue_reference
{
  using type = _Tp;
};

template<typename _Tp>
struct __remove_rvalue_reference<_Tp&&>
{
  using type = _Tp;
};

template<typename _Tp>
using __remove_rvalue_reference_t =
  typename __remove_rvalue_reference<_Tp>::type;

template<
  typename _Awaitable,
  typename _AwaitResult = await_result_t<_Awaitable>,
  std::enable_if_t<std::is_void_v<_AwaitResult>, int> = 0>
__sync_wait_task<_AwaitResult> __make_sync_wait_task(_Awaitable&& __awaitable)
{
  co_await static_cast<_Awaitable&&>(__awaitable);
}

template<
  typename _Awaitable,
  typename _AwaitResult = await_result_t<_Awaitable>,
  std::enable_if_t<!std::is_void_v<_AwaitResult>, int> = 0>
__sync_wait_task<_AwaitResult> __make_sync_wait_task(_Awaitable&& __awaitable)
{
  co_yield co_await static_cast<_Awaitable&&>(__awaitable);
}

template<typename _Awaitable>
auto sync_wait(_Awaitable&& __awaitable)
  -> __remove_rvalue_reference_t<await_result_t<_Awaitable>>
{
  return _VSTD_CORO::__make_sync_wait_task(
    static_cast<_Awaitable&&>(__awaitable)).get();
}

_LIBCPP_END_NAMESPACE_EXPERIMENTAL_COROUTINES

#endif
