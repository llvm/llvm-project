//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T, class U>
// concept invocable;

#include <chrono>
#include <concepts>
#include <memory>
#include <random>
#include <type_traits>

#include "../functions.h"

// clang-format off
template <class F, class... Args>
requires std::invocable<F, Args...>
constexpr void ModelsInvocable(F, Args&&...) noexcept{}

template <class F, class... Args>
requires(!std::invocable<F, Args...>)
constexpr void NotInvocable(F, Args&&...) noexcept {}
// clang-format on

static_assert(!std::invocable<void>);
static_assert(!std::invocable<void*>);
static_assert(!std::invocable<int>);
static_assert(!std::invocable<int&>);
static_assert(!std::invocable<int&&>);

int main(int, char**) {
  {
    using namespace RegularInvocable;

    ModelsInvocable(F);
    NotInvocable(F, 0);

    ModelsInvocable(G, 2);
    NotInvocable(G);
    NotInvocable(G, 3, 0);

    NotInvocable(&A::I);
    NotInvocable(&A::F);

    {
      A X;
      ModelsInvocable(&A::I, X);
      ModelsInvocable(&A::F, X);
      ModelsInvocable(&A::G, X, 0);
      NotInvocable(&A::G, X);
      NotInvocable(&A::G, 0);
      NotInvocable(&A::H);

      A const& Y = X;
      ModelsInvocable(&A::I, Y);
      ModelsInvocable(&A::F, Y);
      NotInvocable(&A::G, Y, 0);
      NotInvocable(&A::H, Y, 0);
    }

    ModelsInvocable(&A::I, A{});
    ModelsInvocable(&A::F, A{});
    ModelsInvocable(&A::G, A{}, 0);
    ModelsInvocable(&A::H, A{}, 0);

    {
      auto Up = std::make_unique<A>();
      ModelsInvocable(&A::I, Up);
      ModelsInvocable(&A::F, Up);
      ModelsInvocable(&A::G, Up, 0);
      NotInvocable(&A::H, Up, 0);
    }
    {
      auto Sp = std::make_shared<A>();
      ModelsInvocable(&A::I, Sp);
      ModelsInvocable(&A::F, Sp);
      ModelsInvocable(&A::G, Sp, 0);
      NotInvocable(&A::H, Sp, 0);
    }
  }
  {
    using namespace Predicate;
    {
      ModelsInvocable(L2rSorted{}, 0, 1, 2);
      NotInvocable(L2rSorted{});
      NotInvocable(L2rSorted{}, 0);
      NotInvocable(L2rSorted{}, 0, 1);
    }
    {
      auto Up = std::make_unique<L2rSorted>();
      ModelsInvocable(&L2rSorted::operator()<int>, Up, 0, 1, 2);
      NotInvocable(&L2rSorted::operator()<int>, Up);
      NotInvocable(&L2rSorted::operator()<int>, Up, 0);
      NotInvocable(&L2rSorted::operator()<int>, Up, 0, 1);
    }
    {
      auto Sp = std::make_shared<L2rSorted>();
      ModelsInvocable(&L2rSorted::operator()<int>, Sp, 0, 1, 2);
      NotInvocable(&L2rSorted::operator()<int>, Sp);
      NotInvocable(&L2rSorted::operator()<int>, Sp, 0);
      NotInvocable(&L2rSorted::operator()<int>, Sp, 0, 1);
    }
  }
  {
    auto G = std::mt19937_64(
        std::chrono::high_resolution_clock().now().time_since_epoch().count());
    auto D = std::uniform_int_distribution<>();
    ModelsInvocable(D, G);
  }
  return 0;
}
