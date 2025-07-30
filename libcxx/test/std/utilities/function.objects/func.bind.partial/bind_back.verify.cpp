//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <functional>

// template<class F, class... Args>
//   constexpr unspecified bind_back(F&& f, Args&&... args);

#include <functional>

#include "types.h"

constexpr int pass(int n) { return n; }

void test() {
  { // Test calling constexpr function from non-constexpr `bind_back` result
    auto f1 = std::bind_back(pass, 1);
    static_assert(f1() == 1); // expected-error {{static assertion expression is not an integral constant expression}}
  }

  { // Test calling `bind_back` with template function
    auto f1 = std::bind_back(do_nothing, 2);
    // expected-error@-1 {{no matching function for call to 'bind_back'}}
  }

  { // Mandates: is_constructible_v<decay_t<F>, F>
    struct F {
      F()         = default;
      F(const F&) = default;
      F(F&)       = delete;

      void operator()() {}
    };

    F f;
    auto f1 = std::bind_back(f);
    // expected-error-re@*:* {{static assertion failed{{.*}}bind_back requires decay_t<F> to be constructible from F}}
  }

  { // Mandates: is_move_constructible_v<decay_t<F>>
    struct F {
      F()         = default;
      F(const F&) = default;
      F(F&&)      = delete;

      void operator()() {}
    };

    F f;
    auto f1 = std::bind_back(f);
    // expected-error-re@*:* {{static assertion failed{{.*}}bind_back requires decay_t<F> to be move constructible}}
  }

  { // Mandates: (is_constructible_v<decay_t<Args>, Args> && ...)
    struct Arg {
      Arg()           = default;
      Arg(const Arg&) = default;
      Arg(Arg&)       = delete;
    };

    Arg x;
    auto f = std::bind_back([](const Arg&) {}, x);
    // expected-error-re@*:* {{static assertion failed{{.*}}bind_back requires all decay_t<Args> to be constructible from respective Args}}
    // expected-error@*:* {{no matching constructor for initialization}}
  }

  { // Mandates: (is_move_constructible_v<decay_t<Args>> && ...)
    struct Arg {
      Arg()           = default;
      Arg(const Arg&) = default;
      Arg(Arg&&)      = delete;
    };

    Arg x;
    auto f = std::bind_back([](Arg&) {}, x);
    // expected-error-re@*:* {{static assertion failed{{.*}}bind_back requires all decay_t<Args> to be move constructible}}
  }
}
