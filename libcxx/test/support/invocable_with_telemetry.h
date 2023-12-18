//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_INVOCABLE_WITH_TELEMETRY_H
#define TEST_SUPPORT_INVOCABLE_WITH_TELEMETRY_H

#include <cassert>
#include <concepts>
#include <utility>

#if TEST_STD_VER < 20
#  error invocable_with_telemetry requires C++20
#else
template <class F>
class invocable_with_telemetry {
public:
  invocable_with_telemetry() = default;

  constexpr invocable_with_telemetry(F f, int& invocations, int& moves, int& copies)
      : f_(f), invocations_(&invocations), moves_(&moves), copies_(&copies) {}

  constexpr invocable_with_telemetry(invocable_with_telemetry&& other)
    requires std::move_constructible<F>
      : f_(std::move(other.f_)),
        invocations_(assert(other.invocations_ != nullptr), std::exchange(other.invocations_, nullptr)),
        moves_(assert(other.moves_ != nullptr), std::exchange(other.moves_, nullptr)),
        copies_(assert(other.copies_ != nullptr), std::exchange(other.copies_, nullptr)) {
    ++*moves_;
  }

  constexpr invocable_with_telemetry(invocable_with_telemetry const& other)
    requires std::copy_constructible<F>
      : f_(other.f_),
        invocations_((assert(other.invocations_ != nullptr), other.invocations_)),
        moves_((assert(other.moves_ != nullptr), other.moves_)),
        copies_((assert(other.copies_ != nullptr), other.copies_)) {
    ++*copies_;
  }

  constexpr invocable_with_telemetry& operator=(invocable_with_telemetry&& other)
    requires std::movable<F>
  {
    // Not using move-and-swap idiom to ensure that copies and moves remain accurate.
    assert(&other != this);
    assert(other.invocations_ != nullptr);
    assert(other.moves_ != nullptr);
    assert(other.copies_ != nullptr);

    f_           = std::move(other.f_);
    invocations_ = std::exchange(other.invocations_, nullptr);
    moves_       = std::exchange(other.moves_, nullptr);
    copies_      = std::exchange(other.copies_, nullptr);

    ++*moves_;
    return *this;
  }

  constexpr invocable_with_telemetry& operator=(invocable_with_telemetry const& other)
    requires std::copyable<F>
  {
    // Not using copy-and-swap idiom to ensure that copies and moves remain accurate.
    assert(&other != this);
    assert(other.invocations_ != nullptr);
    assert(other.moves_ != nullptr);
    assert(other.copies_ != nullptr);

    f_           = other.f_;
    invocations_ = other.invocations_;
    moves_       = other.moves_;
    copies_      = other.copies_;

    ++*copies_;
    return *this;
  }

  template <class... Args>
    requires std::invocable<F&, Args...>
  constexpr decltype(auto) operator()(Args&&... args) noexcept(std::is_nothrow_invocable_v<F&, Args...>) {
    assert(invocations_);
    ++*invocations_;
    return f_(std::forward<Args>(args)...);
  }

private:
  F f_              = F();
  int* invocations_ = nullptr;
  int* moves_       = nullptr;
  int* copies_      = nullptr;
};

template <class F>
invocable_with_telemetry(F f, int& invocations, int& moves, int& copies) -> invocable_with_telemetry<F>;

#endif // TEST_STD_VER < 20
#endif // TEST_SUPPORT_INVOCABLE_WITH_TELEMETRY_H
