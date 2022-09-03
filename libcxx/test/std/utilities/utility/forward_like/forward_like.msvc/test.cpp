// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include <type_traits>
#include <utility>

using namespace std;

struct U {}; // class type so const-qualification is not stripped from a prvalue
using CU = const U;
using T  = int;
using CT = const T;

U u{};
const U& cu = u;

static_assert(is_same_v<decltype(forward_like<T>(U{})), U&&>);
static_assert(is_same_v<decltype(forward_like<T>(CU{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<T>(u)), U&&>);
static_assert(is_same_v<decltype(forward_like<T>(cu)), CU&&>);
static_assert(is_same_v<decltype(forward_like<T>(std::move(u))), U&&>);
static_assert(is_same_v<decltype(forward_like<T>(std::move(cu))), CU&&>);

static_assert(is_same_v<decltype(forward_like<CT>(U{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT>(CU{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT>(u)), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT>(cu)), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT>(std::move(u))), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT>(std::move(cu))), CU&&>);

static_assert(is_same_v<decltype(forward_like<T&>(U{})), U&>);
static_assert(is_same_v<decltype(forward_like<T&>(CU{})), CU&>);
static_assert(is_same_v<decltype(forward_like<T&>(u)), U&>);
static_assert(is_same_v<decltype(forward_like<T&>(cu)), CU&>);
static_assert(is_same_v<decltype(forward_like<T&>(std::move(u))), U&>);
static_assert(is_same_v<decltype(forward_like<T&>(std::move(cu))), CU&>);

static_assert(is_same_v<decltype(forward_like<CT&>(U{})), CU&>);
static_assert(is_same_v<decltype(forward_like<CT&>(CU{})), CU&>);
static_assert(is_same_v<decltype(forward_like<CT&>(u)), CU&>);
static_assert(is_same_v<decltype(forward_like<CT&>(cu)), CU&>);
static_assert(is_same_v<decltype(forward_like<CT&>(std::move(u))), CU&>);
static_assert(is_same_v<decltype(forward_like<CT&>(std::move(cu))), CU&>);

static_assert(is_same_v<decltype(forward_like<T&&>(U{})), U&&>);
static_assert(is_same_v<decltype(forward_like<T&&>(CU{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<T&&>(u)), U&&>);
static_assert(is_same_v<decltype(forward_like<T&&>(cu)), CU&&>);
static_assert(is_same_v<decltype(forward_like<T&&>(std::move(u))), U&&>);
static_assert(is_same_v<decltype(forward_like<T&&>(std::move(cu))), CU&&>);

static_assert(is_same_v<decltype(forward_like<CT&&>(U{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT&&>(CU{})), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT&&>(u)), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT&&>(cu)), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT&&>(std::move(u))), CU&&>);
static_assert(is_same_v<decltype(forward_like<CT&&>(std::move(cu))), CU&&>);

static_assert(noexcept(forward_like<T>(u)));

static_assert(is_same_v<decltype(forward_like<U&>(u)), U&>);
static_assert(is_same_v<decltype(forward_like<CU&>(cu)), CU&>);
static_assert(is_same_v<decltype(forward_like<U&&>(std::move(u))), U&&>);
static_assert(is_same_v<decltype(forward_like<CU&&>(std::move(cu))), CU&&>);

struct NoCtorCopyMove {
  NoCtorCopyMove() = delete;
  NoCtorCopyMove(const NoCtorCopyMove&) = delete;
  NoCtorCopyMove(NoCtorCopyMove&&) = delete;
};

static_assert(is_same_v<decltype(forward_like<CT&&>(declval<NoCtorCopyMove>())), const NoCtorCopyMove&&>);
static_assert(is_same_v<decltype(forward_like<CT&>(declval<NoCtorCopyMove>())), const NoCtorCopyMove&>);
static_assert(is_same_v<decltype(forward_like<T&&>(declval<NoCtorCopyMove>())), NoCtorCopyMove&&>);
static_assert(is_same_v<decltype(forward_like<T&>(declval<NoCtorCopyMove>())), NoCtorCopyMove&>);

static_assert(noexcept(forward_like<T>(declval<NoCtorCopyMove>())));

constexpr bool test() {
  {
    int val       = 1729;
    auto&& result = forward_like<const double&>(val);
    static_assert(is_same_v<decltype(result), const int&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = forward_like<double&>(val);
    static_assert(is_same_v<decltype(result), int&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = forward_like<const double&&>(val);
    static_assert(is_same_v<decltype(result), const int&&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = forward_like<double&&>(val);
    static_assert(is_same_v<decltype(result), int&&>);
    assert(&result == &val);
  }
  return true;
}

int main() {
  assert(test());
  static_assert(test());
}
