//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <future>

// class packaged_task<R(ArgTypes...)>git
// future<R> get_future();
// void operator()(ArgTypes... args);
// void make_ready_at_thread_exit(ArgTypes... args);

// This test is designed to cause and allow TSAN to detect the race condition

#include <cassert>
#include <chrono>
#include <future>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

void delay() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }

void test_operator() {
  std::packaged_task<int()> p([] {
    delay();
    return 42;
  });

  std::thread t      = support::make_test_thread([&p] { p(); });
  std::future<int> f = p.get_future();

  assert(f.get() == 42);
  t.join();
}

void test_operator_void() {
  bool ran = false;
  std::packaged_task<void()> p([&ran] {
    delay();
    ran = true;
  });

  std::thread t       = support::make_test_thread([&p] { p(); });
  std::future<void> f = p.get_future();

  f.get();
  t.join();
  assert(ran);
}

void test_make_ready_at_thread_exit() {
  std::packaged_task<int()> p([] {
    delay();
    return 42;
  });

  std::thread t      = support::make_test_thread([&p] { p.make_ready_at_thread_exit(); });
  std::future<int> f = p.get_future();

  assert(f.get() == 42);
  t.join();
}

void test_make_ready_at_thread_exit_void() {
  bool ran = false;
  std::packaged_task<void()> p([&ran] {
    delay();
    ran = true;
  });

  std::thread t       = support::make_test_thread([&p] { p.make_ready_at_thread_exit(); });
  std::future<void> f = p.get_future();

  f.get();
  t.join();
  assert(ran);
}

int main(int, char**) {
  for (int i = 0; i < 25; ++i) {
    test_operator();
    test_operator_void();
    test_make_ready_at_thread_exit();
    test_make_ready_at_thread_exit_void();
  }

  return 0;
}
