//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-execution

// <execution>
//
// namespace std::execution {
//   struct set_value_t {
//     template<class Receiver, class... Args>
//       requires (!is_lvalue_reference_v<Receiver> && !is_const_v<Receiver>) &&
//                requires(Receiver&& rcvr, Args&&... args) {
//                  std::forward<Receiver>(rcvr).set_value(std::forward<Args>(args)...);
//                }
//     constexpr void operator()(Receiver&& rcvr, Args&&... args) const noexcept;
//   };
//
//   struct set_error_t {
//     template<class Receiver, class Error>
//       requires (!is_lvalue_reference_v<Receiver> && !is_const_v<Receiver>) &&
//                requires(Receiver&& rcvr, Error&& error) {
//                  std::forward<Receiver>(rcvr).set_error(std::forward<Error>(error));
//                }
//     constexpr void operator()(Receiver&& rcvr, Error&& error) const noexcept;
//   };
//
//   struct set_stopped_t {
//     template<class Receiver>
//       requires (!is_lvalue_reference_v<Receiver> && !is_const_v<Receiver>) &&
//                requires(Receiver&& rcvr) {
//                  std::forward<Receiver>(rcvr).set_stopped();
//                }
//     constexpr void operator()(Receiver&& rcvr) const noexcept;
//   };
//
//   inline constexpr set_value_t set_value{};
//   inline constexpr set_error_t set_error{};
//   inline constexpr set_stopped_t set_stopped{};
// }

#include <cassert>
#include <execution>
#include <type_traits>
#include <utility>

namespace ex = std::execution;

static_assert(std::is_same_v<decltype(ex::set_value), const ex::set_value_t>);
static_assert(std::is_same_v<decltype(ex::set_error), const ex::set_error_t>);
static_assert(std::is_same_v<decltype(ex::set_stopped), const ex::set_stopped_t>);

// Reject receivers that have no set_xxx members.
struct NoMemberReceiver {};
static_assert(!std::is_invocable_v<ex::set_value_t, NoMemberReceiver, int>);
static_assert(!std::is_invocable_v<ex::set_error_t, NoMemberReceiver, int>);
static_assert(!std::is_invocable_v<ex::set_stopped_t, NoMemberReceiver>);

struct CvrefTestReceiver {
  void set_value(int) const noexcept;
  void set_error(int) const noexcept;
  void set_stopped() const noexcept;
};

static_assert(std::is_nothrow_invocable_v<ex::set_value_t, CvrefTestReceiver, int>);
static_assert(std::is_nothrow_invocable_v<ex::set_error_t, CvrefTestReceiver, int>);
static_assert(std::is_nothrow_invocable_v<ex::set_stopped_t, CvrefTestReceiver>);

static_assert(!std::is_invocable_v<ex::set_value_t, CvrefTestReceiver&, int>);
static_assert(!std::is_invocable_v<ex::set_error_t, CvrefTestReceiver&, int>);
static_assert(!std::is_invocable_v<ex::set_stopped_t, CvrefTestReceiver&>);

static_assert(!std::is_invocable_v<ex::set_value_t, const CvrefTestReceiver, int>);
static_assert(!std::is_invocable_v<ex::set_error_t, const CvrefTestReceiver, int>);
static_assert(!std::is_invocable_v<ex::set_stopped_t, const CvrefTestReceiver>);

struct StaticReceiver {
  static void set_value(int) noexcept;
  static void set_error(int) noexcept;
  static void set_stopped() noexcept;
};

static_assert(std::is_nothrow_invocable_v<ex::set_value_t, StaticReceiver, int>);
static_assert(std::is_nothrow_invocable_v<ex::set_error_t, StaticReceiver, int>);
static_assert(std::is_nothrow_invocable_v<ex::set_stopped_t, StaticReceiver>);

struct RvalueReceiver {
  int& result;

  constexpr void set_value() && noexcept { result = 0; }
  constexpr void set_value(int value) && noexcept { result = value; }
  constexpr void set_value(int& output, int&& value) && noexcept {
    output = value;
    result = value;
  }
  constexpr void set_error(int& error) && noexcept { result = error; }
  constexpr void set_error(int&& error) && noexcept { result = error; }
  constexpr void set_stopped() && noexcept { result = 0; }
};

// This receiver accepts at most two values, one error, and no stopped arguments.
static_assert(!std::is_invocable_v<ex::set_value_t, RvalueReceiver, int, int, int>);
static_assert(!std::is_invocable_v<ex::set_error_t, RvalueReceiver>);
static_assert(!std::is_invocable_v<ex::set_error_t, RvalueReceiver, int, int>);
static_assert(!std::is_invocable_v<ex::set_stopped_t, RvalueReceiver, int>);

constexpr bool test() {
  int result = -1;
  RvalueReceiver receiver{result};
  ex::set_value(std::move(receiver));
  assert(result == 0);

  result = -1;
  ex::set_value(RvalueReceiver{result}, 42);
  assert(result == 42);

  // Forward multiple arguments, preserving lvalue and rvalue references.
  int output = 0;
  result     = -1;
  ex::set_value(RvalueReceiver{result}, output, 42);
  assert(output == 42);
  assert(result == 42);

  result = -1;
  ex::set_error(RvalueReceiver{result}, 42);
  assert(result == 42);

  int error = 42;
  result    = -1;
  ex::set_error(RvalueReceiver{result}, error);
  assert(result == 42);

  result = -1;
  ex::set_stopped(RvalueReceiver{result});
  assert(result == 0);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
