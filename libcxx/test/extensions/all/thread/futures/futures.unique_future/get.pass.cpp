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
// UNSUPPORTED: no-exceptions

// <future>

// class future<R>

// R future::get();
// R& future<R&>::get();
// void future<void>::get();

#include <future>
#include <cassert>

int main(int, char**) {
  {
    std::future<int> f;
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }
  {
    std::promise<int> p;
    std::future<int> f = p.get_future();
    p.set_value(3);
    (void)f.get();
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }
  {
    std::future<int&> f;
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }
  {
    std::promise<int&> p;
    std::future<int&> f = p.get_future();
    int j_val           = 5;
    p.set_value(j_val);
    (void)f.get();
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }
  {
    std::future<void> f;
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }
  {
    std::promise<void> p;
    std::future<void> f = p.get_future();
    p.set_value();
    (void)f.get();
    try {
      (void)f.get();
      assert(false);
    } catch (const std::future_error& e) {
      assert(e.code() == std::make_error_code(std::future_errc::no_state));
    }
  }

  return 0;
}
