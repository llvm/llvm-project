//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] bool joinable() const noexcept;
// [[nodiscard]] id get_id() const noexcept;
// [[nodiscard]] native_handle_type native_handle();
// [[nodiscard]] stop_source get_stop_source() noexcept;
// [[nodiscard]] stop_token get_stop_token() const noexcept;
// [[nodiscard]] static unsigned int hardware_concurrency() noexcept;

#include <thread>

void test() {
  std::jthread jt;
  jt.joinable();             // expected-warning {{ignoring return value of function}}
  jt.get_id();               // expected-warning {{ignoring return value of function}}
  jt.native_handle();        // expected-warning {{ignoring return value of function}}
  jt.get_stop_source();      // expected-warning {{ignoring return value of function}}
  jt.get_stop_token();       // expected-warning {{ignoring return value of function}}
  jt.hardware_concurrency(); // expected-warning {{ignoring return value of function}}
}
