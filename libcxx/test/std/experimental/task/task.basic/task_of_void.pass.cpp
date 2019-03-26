// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <experimental/task>
#include "../manual_reset_event.hpp"
#include "../sync_wait.hpp"

#include <optional>
#include <thread>

namespace coro = std::experimental::coroutines_v1;

static bool has_f_executed = false;

static coro::task<void> f()
{
  has_f_executed = true;
  co_return;
}

static void test_coroutine_executes_lazily()
{
  coro::task<void> t = f();
  assert(!has_f_executed);
  coro::sync_wait(t);
  assert(has_f_executed);
}

static std::optional<int> last_value_passed_to_g;

static coro::task<void> g(int a)
{
  last_value_passed_to_g = a;
  co_return;
}

void test_coroutine_accepts_arguments()
{
  auto t = g(123);
  assert(!last_value_passed_to_g);
  coro::sync_wait(t);
  assert(last_value_passed_to_g);
  assert(*last_value_passed_to_g == 123);
}

int shared_value = 0;
int read_value = 0;

coro::task<void> consume_async(manual_reset_event& event)
{
  co_await event;
  read_value = shared_value;
}

void produce(manual_reset_event& event)
{
  shared_value = 101;
  event.set();
}

void test_async_completion()
{
  manual_reset_event e;
  std::thread t1{ [&e]
  {
    sync_wait(consume_async(e));
  }};

  assert(read_value == 0);

  std::thread t2{ [&e] { produce(e); }};

  t1.join();

  assert(read_value == 101);

  t2.join();
}

int main()
{
  test_coroutine_executes_lazily();
  test_coroutine_accepts_arguments();
  test_async_completion();

  return 0;
}
