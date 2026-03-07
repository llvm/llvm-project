//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <functional>

// template<auto f, class... Args>
//   constexpr unspecified bind_front(Args&&...);
// Mandates:
// - (is_constructible_v<BoundArgs, Args> && ...) is true, and
// - (is_move_constructible_v<BoundArgs> && ...) is true, and
// - If is_pointer_v<F> || is_member_pointer_v<F> is true, then f != nullptr is true.

#include <functional>

struct AnyArgs {
  template <class... Args>
  void operator()(Args&&...) {}
};

void test() {
  { // (is_constructible_v<BoundArgs, Args> && ...) is true
    struct Arg {
      Arg()           = default;
      Arg(const Arg&) = default;
      Arg(Arg&)       = delete;
    };

    Arg arg;
    auto _ = std::bind_front<AnyArgs{}>(arg);
    // expected-error@*:* {{static assertion failed due to requirement 'is_constructible_v<Arg, Arg &>': bind_front requires all decay_t<Args> to be constructible from respective Args}}
    // expected-error@*:* 0-1{{call to deleted constructor of 'Arg'}}
  }

  { // (is_move_constructible_v<BoundArgs> && ...) is true
    struct Arg {
      Arg()           = default;
      Arg(Arg&&)      = delete;
      Arg(const Arg&) = default;
    };

    Arg arg;
    auto _ = std::bind_front<AnyArgs{}>(arg);
    // expected-error@*:* {{static assertion failed due to requirement 'is_move_constructible_v<Arg>': bind_front requires all decay_t<Args> to be move constructible}}
  }

  { // If is_pointer_v<F> || is_member_pointer_v<F> is true, then f != nullptr is true
    struct X {};

    auto _ = std::bind_front<static_cast<void (*)()>(nullptr)>();
    // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': bind_front: f cannot be equal to nullptr}}

    auto _ = std::bind_front<static_cast<int X::*>(nullptr)>();
    // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': bind_front: f cannot be equal to nullptr}}

    auto _ = std::bind_front<static_cast<void (X::*)()>(nullptr)>();
    // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': bind_front: f cannot be equal to nullptr}}
  }
}
