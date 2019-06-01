// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <experimental/coroutine>
#include <cassert>

#include "test_macros.h"
using namespace std::experimental;

struct error {};

template <typename T, typename Error = int>
struct expected {

  struct Data {
    T val;
    Error error;
  };
  Data data;

  struct DataPtr {
    Data *p;
    ~DataPtr() { delete p; }
  };

  expected() {}
  expected(T val) : data{std::move(val),{}} {}
  expected(struct error, Error error) : data{{}, std::move(error)} {}
  expected(DataPtr & p) : data{std::move(p.p->val), std::move(p.p->error)} {}

  struct promise_type {
    Data* data;
    DataPtr get_return_object() { data = new Data{}; return {data}; }
    suspend_never initial_suspend() { return {}; }
    suspend_never final_suspend() { return {}; }
    void return_value(T v) { data->val = std::move(v); data->error = {};}
    void unhandled_exception() {}
  };

  bool await_ready() { return !data.error; }
  T await_resume() { return std::move(data.val); }
  void await_suspend(coroutine_handle<promise_type> h) {
    h.promise().data->error =std::move(data.error);
    h.destroy();
  }

  T const& value() { return data.val; }
  Error const& error() { return data.error; }
};

expected<int> g() { return {0}; }
expected<int> h() { return {error{}, 42}; }

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
