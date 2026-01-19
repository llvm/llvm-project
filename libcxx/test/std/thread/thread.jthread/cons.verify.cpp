//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++20

// <jthread>

// class jthread

// template<class F, class... Args>
// explicit jthread(F&& f, Args&&... args);

#include <thread>

void test() {
  { // Mandates: is_constructible_v<decay_t<F>, F>
    struct F {
      F()         = default;
      F(const F&) = delete;
      F(F&&)      = default;

      void operator()() {}
    };

    F f;
    std::jthread t(f);
    // expected-error@*:* 2 {{static assertion failed}}
    // expected-error@*:* 0-2 {{call to deleted constructor}}
    // expected-error@*:* {{no matching constructor for initialization}}
  }

  { // Mandates: is_constructible_v<decay_t<F>, F>
    struct F {
      F()         = default;
      F(const F&) = default;
      F(F&&)      = delete;

      void operator()() {}
    };

    std::jthread t(F{});
    // expected-error@*:* 2 {{static assertion failed}}
    // expected-error@*:* 0-2 {{call to deleted constructor}}
    // expected-error@*:* {{no matching constructor for initialization}}
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
    std::jthread t(F{}, arg);
    // expected-error@*:* 2 {{static assertion failed}}
    // expected-error@*:* 0-1 {{call to deleted constructor}}
    // expected-error@*:* {{no matching constructor for initialization}}
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

    std::jthread t(F{}, Arg{});
    // expected-error@*:* 2 {{static assertion failed}}
    // expected-error@*:* 0-1 {{call to deleted constructor}}
    // expected-error@*:* {{no matching constructor for initialization}}
  }

  { // Mandates: is_invocable_v<decay_t<F>, decay_t<Args>...>
    struct F {};

    std::jthread t(F{});
    // expected-error@*:* 2 {{static assertion failed}}
    // expected-error@*:* 0-1 {{no matching function for call to '__invoke'}}
    // expected-error@*:* 0-1 {{attempt to use a deleted function}}
  }
}
