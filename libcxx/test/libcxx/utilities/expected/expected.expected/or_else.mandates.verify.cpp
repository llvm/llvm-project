//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test the mandates
// template<class F> constexpr auto or_else(F&& f) &;
// Mandates:
//  Let G be std::remove_cvref_t<std::invoke_result<F, decltype(error())>>
//  G is a specialization of std::expected and std::is_same_v<G:value_type, T> is true

// template<class F> constexpr auto or_else(F&& f) const &;
// Mandates:
//  Let G be std::remove_cvref_t<std::invoke_result<F, decltype(error())>>
//  G is a specialization of std::expected and std::is_same_v<G:value_type, T> is true

// template<class F> constexpr auto or_else(F&& f) &&;
// Mandates:
//  Let G be std::remove_cvref_t<std::invoke_result<F, decltype(error())>>
//  G is a specialization of std::expected and std::is_same_v<G:value_type, T> is true

// template<class F> constexpr auto or_else(F&& f) const &&;
// Mandates:
//  Let G be std::remove_cvref_t<std::invoke_result<F, decltype(error())>>
//  G is a specialization of std::expected and std::is_same_v<G:value_type, T> is true

#include <expected>
#include <utility>

struct NotSameAsInt {};

int lval_return_not_std_expected(int&) { return 0; }
int clval_return_not_std_expected(const int&) { return 0; }
int rval_return_not_std_expected(int&&) { return 0; }
int crval_return_not_std_expected(const int&&) { return 0; }

std::expected<NotSameAsInt, int> lval_error_type_not_same_as_int(int&) { return {}; }
std::expected<NotSameAsInt, int> clval_error_type_not_same_as_int(const int&) { return {}; }
std::expected<NotSameAsInt, int> rval_error_type_not_same_as_int(int&&) { return {}; }
std::expected<NotSameAsInt, int> crval_error_type_not_same_as_int(const int&&) { return {}; }

// clang-format off
void test() {
  // Test & overload
  {
    // G is not a specialization of std::expected
    {
      std::expected<int, int> f1(std::unexpected<int>(1));
      f1.or_else(lval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<int (&)(int &)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(error()) must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<G:value_type, T>
    {
      std::expected<int, int> f1(std::unexpected<int>(1));
      f1.or_else(lval_error_type_not_same_as_int);  // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<std::expected<NotSameAsInt, int> (&)(int &)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(error()) must have the same value_type as this expected}}
    }
  }

  // Test const& overload
  {
    // G is not a specialization of std::expected
    {
      const std::expected<int, int> f1(std::unexpected<int>(1));
      f1.or_else(clval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<int (&)(const int &)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(error()) must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<G:value_type, T>
    {
      const std::expected<int, int> f1(std::unexpected<int>(1));
      f1.or_else(clval_error_type_not_same_as_int);  // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<std::expected<NotSameAsInt, int> (&)(const int &)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(error()) must have the same value_type as this expected}}
    }
  }

  // Test && overload
  {
    // G is not a specialization of std::expected
    {
      std::expected<int, int> f1(std::unexpected<int>(1));
      std::move(f1).or_else(rval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<int (&)(int &&)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(std::move(error())) must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<G:value_type, T>
    {
      std::expected<int, int> f1(std::unexpected<int>(1));
      std::move(f1).or_else(rval_error_type_not_same_as_int); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<std::expected<NotSameAsInt, int> (&)(int &&)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(std::move(error())) must have the same value_type as this expected}}
    }
  }

  // Test const&& overload
  {
    // G is not a specialization of std::expected
    {
      const std::expected<int, int> f1(std::unexpected<int>(1));
      std::move(f1).or_else(crval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<int (&)(const int &&)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(std::move(error())) must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<G:value_type, T>
    {
      const std::expected<int, int> f1(std::unexpected<int>(1));
      std::move(f1).or_else(crval_error_type_not_same_as_int); // expected-note{{in instantiation of function template specialization 'std::expected<int, int>::or_else<std::expected<NotSameAsInt, int> (&)(const int &&)>' requested here}}
      // expected-error-re@*:* {{static assertion failed {{.*}}The result of f(std::move(error())) must have the same value_type as this expected}}
    }
  }
}
// clang-format on
