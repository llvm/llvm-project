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

// template <class Rep, class Period>
//   future_status
//   wait_for(const chrono::duration<Rep, Period>& rel_time) const;

#include <future>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

typedef std::chrono::milliseconds ms;

static const ms sleepTime(500);
static const ms waitTime(5000);

void func1(std::promise<int> p)
{
  std::this_thread::sleep_for(sleepTime);
  p.set_value(3);
}

int j = 0;

void func3(std::promise<int&> p)
{
  std::this_thread::sleep_for(sleepTime);
  j = 5;
  p.set_value(j);
}

void func5(std::promise<void> p)
{
  std::this_thread::sleep_for(sleepTime);
  p.set_value();
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
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    assert(f.wait_for(waitTime) == std::future_status::ready);
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
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    assert(f.wait_for(waitTime) == std::future_status::ready);
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
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    assert(f.wait_for(waitTime) == std::future_status::ready);
    assert(f.valid());
    f.wait();
    assert(f.valid());
  }

  {
    typedef int T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    Clock::time_point t0 = Clock::now();
    support::make_test_thread(func1, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    f.wait();
    Clock::time_point t1 = Clock::now();
    assert(f.valid());
    assert(t1 - t0 >= sleepTime);
    assert(f.wait_for(waitTime) == std::future_status::ready);
    assert(f.valid());
  }
  {
    typedef int& T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    Clock::time_point t0 = Clock::now();
    support::make_test_thread(func3, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    f.wait();
    Clock::time_point t1 = Clock::now();
    assert(f.valid());
    assert(t1 - t0 >= sleepTime);
    assert(f.wait_for(waitTime) == std::future_status::ready);
    assert(f.valid());
  }
  {
    typedef void T;
    std::promise<T> p;
    std::shared_future<T> f = p.get_future();
    Clock::time_point t0 = Clock::now();
    support::make_test_thread(func5, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_for(ms(1)) == std::future_status::timeout);
    assert(f.valid());
    f.wait();
    Clock::time_point t1 = Clock::now();
    assert(f.valid());
    assert(t1 - t0 >= sleepTime);
    assert(f.wait_for(waitTime) == std::future_status::ready);
    assert(f.valid());
  }

  return 0;
}
