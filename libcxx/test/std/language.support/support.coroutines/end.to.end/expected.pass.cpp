//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <coroutine>
#include <cassert>
#include <memory>

#include "test_macros.h"

struct error_tag { };

template <typename T, typename Error = int>
struct expected {

  struct Data {
    Data() : val(), error() { }
    Data(T v, Error e) : val(v), error(e) { }
    T val;
    Error error;
  };
  std::shared_ptr<Data> data;

  expected(T val) : data(std::make_shared<Data>(val, Error())) {}
  expected(error_tag, Error error) : data(std::make_shared<Data>(T(), error)) {}
  expected(std::shared_ptr<Data> p) : data(p) {}

  struct promise_type {
    std::shared_ptr<Data> data;
    expected get_return_object() { data = std::make_shared<Data>(); return {data}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_value(T v) { data->val = v; data->error = {}; }
    void unhandled_exception() {}
  };

  bool await_ready() { return !data->error; }
  T await_resume() { return data->val; }
  void await_suspend(std::coroutine_handle<promise_type> h) {
    h.promise().data->error = data->error;
    h.destroy();
  }

  T const& value() { return data->val; }
  Error const& error() { return data->error; }
};

expected<int> g() { return {0}; }
expected<int> h() { return {error_tag{}, 42}; }

extern "C" void print(int);

bool f1_started, f1_resumed = false;
expected<int> f1() {
  f1_started = true;
  (void)(co_await g());
  f1_resumed = true;
  co_return 100;
}

bool f2_started, f2_resumed = false;
expected<int> f2() {
  f2_started = true;
  (void)(co_await h());
  f2_resumed = true;
  co_return 200;
}

int main(int, char**) {
  auto c1 = f1();
  assert(f1_started && f1_resumed);
  assert(c1.value() == 100);
  assert(c1.error() == 0);

  auto c2 = f2();
  assert(f2_started && !f2_resumed);
  assert(c2.value() == 0);
  assert(c2.error() == 42);

  return 0;
}
