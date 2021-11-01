//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

// <future>

// class shared_future<R>

// template <class Clock, class Duration>
//   future_status
//   wait_until(const chrono::time_point<Clock, Duration>& abs_time) const;

#include <future>
#include <atomic>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

enum class WorkerThreadState { Uninitialized, AllowedToRun, Exiting };
typedef std::chrono::milliseconds ms;

std::atomic<WorkerThreadState> thread_state(WorkerThreadState::Uninitialized);

void set_worker_thread_state(WorkerThreadState state)
{
  thread_state.store(state, std::memory_order_relaxed);
}

void wait_for_worker_thread_state(WorkerThreadState state)
{
  while (thread_state.load(std::memory_order_relaxed) != state)
    std::this_thread::yield();
}

void func1(std::promise<int> p)
{
  wait_for_worker_thread_state(WorkerThreadState::AllowedToRun);
  p.set_value(3);
  set_worker_thread_state(WorkerThreadState::Exiting);
}

int j = 0;

void func3(std::promise<int&> p)
{
  wait_for_worker_thread_state(WorkerThreadState::AllowedToRun);
  j = 5;
  p.set_value(j);
  set_worker_thread_state(WorkerThreadState::Exiting);
}

void func5(std::promise<void> p)
{
  wait_for_worker_thread_state(WorkerThreadState::AllowedToRun);
  p.set_value();
  set_worker_thread_state(WorkerThreadState::Exiting);
}

int main(int, char**)
{
  typedef std::chrono::high_resolution_clock Clock;

  {
    typedef int T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    support::make_test_thread(func1, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::timeout);
    assert(f.valid());

    // allow the worker thread to produce the result and wait until the worker is done
    set_worker_thread_state(WorkerThreadState::AllowedToRun);
    wait_for_worker_thread_state(WorkerThreadState::Exiting);

    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::ready);
    assert(f.valid());
    f.wait();
    assert(f.valid());
  }
  {
    typedef int& T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    support::make_test_thread(func3, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::timeout);
    assert(f.valid());

    // allow the worker thread to produce the result and wait until the worker is done
    set_worker_thread_state(WorkerThreadState::AllowedToRun);
    wait_for_worker_thread_state(WorkerThreadState::Exiting);

    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::ready);
    assert(f.valid());
    f.wait();
    assert(f.valid());
  }
  {
    typedef void T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    support::make_test_thread(func5, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::timeout);
    assert(f.valid());

    // allow the worker thread to produce the result and wait until the worker is done
    set_worker_thread_state(WorkerThreadState::AllowedToRun);
    wait_for_worker_thread_state(WorkerThreadState::Exiting);

    assert(f.wait_until(Clock::now() + ms(10)) == std::future_status::ready);
    assert(f.valid());
    f.wait();
    assert(f.valid());
  }

  return 0;
}
