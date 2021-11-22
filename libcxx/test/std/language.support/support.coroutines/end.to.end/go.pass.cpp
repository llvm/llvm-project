//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-coroutines

#include <cassert>
#include <coroutine>
#include <memory>

#include "test_macros.h"

bool cancel = false;

struct goroutine
{
  static int const N = 10;
  static int count;
  static std::coroutine_handle<> stack[N];

  static void schedule(std::coroutine_handle<>& rh)
  {
    assert(count < N);
    stack[count++] = rh;
    rh = nullptr;
  }

  ~goroutine() {}

  static void run_one()
  {
    assert(count > 0);
    stack[--count]();
  }

  struct promise_type
  {
    std::suspend_never initial_suspend() {
      return {};
    }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    goroutine get_return_object() {
      return{};
    }
    void unhandled_exception() {}
  };
};
int goroutine::count;
std::coroutine_handle<> goroutine::stack[N];

std::coroutine_handle<goroutine::promise_type> workaround;

class channel;

struct push_awaiter {
  channel* ch;
  bool await_ready() {return false; }
  void await_suspend(std::coroutine_handle<> rh);
  void await_resume() {}
};

struct pull_awaiter {
  channel * ch;

  bool await_ready();
  void await_suspend(std::coroutine_handle<> rh);
  int await_resume();
};

class channel
{
  using T = int;

  friend struct push_awaiter;
  friend struct pull_awaiter;

  T const* pvalue = nullptr;
  std::coroutine_handle<> reader = nullptr;
  std::coroutine_handle<> writer = nullptr;
public:
  push_awaiter push(T const& value)
  {
    assert(pvalue == nullptr);
    assert(!writer);
    pvalue = &value;

    return { this };
  }

  pull_awaiter pull()
  {
    assert(!reader);

    return { this };
  }

  void sync_push(T const& value)
  {
    assert(!pvalue);
    pvalue = &value;
    assert(reader);
    reader();
    assert(!pvalue);
    reader = nullptr;
  }

  auto sync_pull()
  {
    while (!pvalue) goroutine::run_one();
    auto result = *pvalue;
    pvalue = nullptr;
    if (writer)
    {
      auto wr = writer;
      writer = nullptr;
      wr();
    }
    return result;
  }
};

void push_awaiter::await_suspend(std::coroutine_handle<> rh)
{
  ch->writer = rh;
  if (ch->reader) goroutine::schedule(ch->reader);
}


bool pull_awaiter::await_ready() {
  return !!ch->writer;
}
void pull_awaiter::await_suspend(std::coroutine_handle<> rh) {
  ch->reader = rh;
}
int pull_awaiter::await_resume() {
  auto result = *ch->pvalue;
  ch->pvalue = nullptr;
  if (ch->writer) {
    //goroutine::schedule(ch->writer);
    auto wr = ch->writer;
    ch->writer = nullptr;
    wr();
  }
  return result;
}

goroutine pusher(channel& left, channel& right)
{
  for (;;) {
    auto val = co_await left.pull();
    co_await right.push(val + 1);
  }
}

const int N = 100;
channel c[N + 1];

int main(int, char**) {
  for (int i = 0; i < N; ++i)
    pusher(c[i], c[i + 1]);

  c[0].sync_push(0);
  int result = c[N].sync_pull();

  assert(result == 100);

  return 0;
}
