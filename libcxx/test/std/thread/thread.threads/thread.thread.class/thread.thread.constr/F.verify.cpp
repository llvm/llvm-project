//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// <thread>

// class thread

// template<class F, class... Args>
// explicit thread(F&& f, Args&&... args);

#include <thread>

void test() {
  { // Mandates: is_constructible_v<decay_t<F>, F>
    struct F {
      F()         = default;
      F(const F&) = delete;
      F(F&&)      = delete;

      void operator()() {}
    };

    F f;
    std::thread t(f);
    // expected-error@*:* {{static assertion failed}}
    // expected-error@*:* {{call to deleted constructor}}
    // expected-error@*:* {{no matching constructor for initialization}}
  }

  { // Mandates: (is_constructible_v<decay_t<Args>, Args> && ...)
    struct Arg {
      Arg()           = default;
      Arg(const Arg&) = delete;
      Arg(Arg&&)      = delete;
    };

    struct F {
      void operator()(const Arg&) const {}
    };

    Arg arg;
    std::thread t(F{}, arg);
    // expected-error@*:* {{static assertion failed}}
    // expected-error@*:* 0+ {{call to deleted constructor}}
    // expected-error@*:* 0+ {{no matching constructor for initialization}}
  }

  { // Mandates: is_invocable_v<decay_t<F>, decay_t<Args>...>
    struct F {};

    std::thread t(F{});
    // expected-error@*:* {{static assertion failed}}
    // expected-error@*:* {{no matching function for call to '__invoke'}}
  }
}
