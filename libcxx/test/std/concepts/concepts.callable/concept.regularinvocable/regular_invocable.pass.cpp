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
// concept regular_invocable;

#include <concepts>
#include <memory>
#include <random>
#include <type_traits>

#include "../functions.h"

// clang-format off
template <class F, class... Args>
requires std::regular_invocable<F, Args...>
constexpr void ModelsRegularInvocable(F, Args&&...) noexcept {}

template <class F, class... Args>
requires (!std::regular_invocable<F, Args...>)
constexpr void NotRegularInvocable(F, Args&&...) noexcept {}
// clang-format on

static_assert(!std::regular_invocable<void>);
static_assert(!std::regular_invocable<void*>);
static_assert(!std::regular_invocable<int>);
static_assert(!std::regular_invocable<int&>);
static_assert(!std::regular_invocable<int&&>);

int main(int, char**) {
  {
    using namespace RegularInvocable;

    ModelsRegularInvocable(F);
    NotRegularInvocable(F, 0);

    ModelsRegularInvocable(G, 2);
    NotRegularInvocable(G);
    NotRegularInvocable(G, 3, 0);

    NotRegularInvocable(&A::I);
    NotRegularInvocable(&A::F);

    {
      A X;
      ModelsRegularInvocable(&A::I, X);
      ModelsRegularInvocable(&A::F, X);
      ModelsRegularInvocable(&A::G, X, 0);
      NotRegularInvocable(&A::G, X);
      NotRegularInvocable(&A::G, 0);
      NotRegularInvocable(&A::H);

      A const& Y = X;
      ModelsRegularInvocable(&A::I, Y);
      ModelsRegularInvocable(&A::F, Y);
      NotRegularInvocable(&A::G, Y, 0);
      NotRegularInvocable(&A::H, Y, 0);
    }

    ModelsRegularInvocable(&A::I, A{});
    ModelsRegularInvocable(&A::F, A{});
    ModelsRegularInvocable(&A::G, A{}, 0);
    ModelsRegularInvocable(&A::H, A{}, 0);

    {
      auto Up = std::make_unique<A>();
      ModelsRegularInvocable(&A::I, Up);
      ModelsRegularInvocable(&A::F, Up);
      ModelsRegularInvocable(&A::G, Up, 0);
      NotRegularInvocable(&A::H, Up, 0);
    }
    {
      auto Sp = std::make_shared<A>();
      ModelsRegularInvocable(&A::I, Sp);
      ModelsRegularInvocable(&A::F, Sp);
      ModelsRegularInvocable(&A::G, Sp, 0);
      NotRegularInvocable(&A::H, Sp, 0);
    }
  }
  {
    using namespace Predicate;
    {
      ModelsRegularInvocable(L2rSorted{}, 0, 1, 2);
      NotRegularInvocable(L2rSorted{});
      NotRegularInvocable(L2rSorted{}, 0);
      NotRegularInvocable(L2rSorted{}, 0, 1);
    }
    {
      auto Up = std::make_unique<L2rSorted>();
      ModelsRegularInvocable(&L2rSorted::operator()<int>, Up, 0, 1, 2);
      NotRegularInvocable(&L2rSorted::operator()<int>, Up);
      NotRegularInvocable(&L2rSorted::operator()<int>, Up, 0);
      NotRegularInvocable(&L2rSorted::operator()<int>, Up, 0, 1);
    }
    {
      auto Sp = std::make_shared<L2rSorted>();
      ModelsRegularInvocable(&L2rSorted::operator()<int>, Sp, 0, 1, 2);
      NotRegularInvocable(&L2rSorted::operator()<int>, Sp);
      NotRegularInvocable(&L2rSorted::operator()<int>, Sp, 0);
      NotRegularInvocable(&L2rSorted::operator()<int>, Sp, 0, 1);
    }
  }
  // {
  // RNG doesn't model regular_invocable, left here for documentation
  // 	auto G = std::mt19937_64(std::random_device()());
  // 	auto D = std::uniform_int_distribution<>();
  // 	models_invocable(D, G);
  // }
  return 0;
}
