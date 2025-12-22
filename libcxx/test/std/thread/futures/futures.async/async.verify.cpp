//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <future>

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(F&& f, Args&&... args);

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(launch policy, F&& f, Args&&... args);


#include <future>

int foo (int x) { return x; }

void f() {
    std::async(                    foo, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::async(std::launch::async, foo, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    { // Mandates: is_constructible_v<decay_t<F>, F>
      struct F {
        F()         = default;
        F(const F&) = delete;
        F(F&&)      = default;

        void operator()() {}
      };

      F f;
      static_cast<void>(std::async(f));
      // expected-error@*:* {{static assertion failed}}
      // expected-error@*:* {{no matching constructor for initialization}}
      // expected-error@*:* {{call to deleted constructor}}
    }

    { // Mandates: is_constructible_v<decay_t<F>, F>
      struct F {
        F()         = default;
        F(const F&) = default;
        F(F&&)      = delete;

        void operator()() {}
      };

      static_cast<void>(std::async(F{}));
      // expected-error@*:* {{static assertion failed}}
      // expected-error@*:* {{no matching constructor for initialization}}
      // expected-error@*:* {{call to deleted constructor}}
    }

    { // Mandates: (is_constructible_v<decay_t<Args>, Args> && ...)
      struct Arg {
        Arg()           = default;
        Arg(const Arg&) = delete;
        Arg(Arg&&)      = default;
      };

      struct F {
        void operator()(const Arg&) const {}
      };

      Arg arg;
      static_cast<void>(std::async(F{}, arg));
      // expected-error@*:* {{static assertion failed}}
      // expected-error@*:* {{call to deleted constructor}}
    }

    { // Mandates: (is_constructible_v<decay_t<Args>, Args> && ...)
      struct Arg {
        Arg()           = default;
        Arg(const Arg&) = default;
        Arg(Arg&&)      = delete;
      };

      struct F {
        void operator()(const Arg&) const {}
      };

      static_cast<void>(std::async(F{}, Arg{}));
      // expected-error@*:* {{static assertion failed}}
      // expected-error@*:* {{call to deleted constructor}}
    }

    { // Mandates: is_invocable_v<decay_t<F>, decay_t<Args>...>
      struct F {};

      static_cast<void>(std::async(F{}));
      // expected-error@*:* {{no matching function}}
    }
}
