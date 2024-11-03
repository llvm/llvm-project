//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test the mandates
// template<class F> constexpr auto and_then(F&& f) &;
// Mandates:
//  Let U be std::remove_cvref_t<std::invoke_result<F>>
//  U is a specialization of std::expected and std::is_same_v<U:error_type, E> is true

// template<class F> constexpr auto and_then(F&& f) const &;
// Mandates:
//  Let U be std::remove_cvref_t<std::invoke_result<F>>
//  U is a specialization of std::expected and std::is_same_v<U:error_type, E> is true

// template<class F> constexpr auto and_then(F&& f) &&;
// Mandates:
//  Let U be std::remove_cvref_t<std::invoke_result<F>>
//  U is a specialization of std::expected and std::is_same_v<U:error_type, E> is true

// template<class F> constexpr auto and_then(F&& f) const &&;
// Mandates:
//  Let U be std::remove_cvref_t<std::invoke_result<F>>
//  U is a specialization of std::expected and std::is_same_v<U:error_type, E> is true

#include <expected>
#include <utility>

struct NotSameAsInt {};

int lval_return_not_std_expected(void) { return 0; }
int clval_return_not_std_expected(void) { return 0; }
int rval_return_not_std_expected(void) { return 0; }
int crval_return_not_std_expected(void) { return 0; }

std::expected<int, NotSameAsInt> lval_error_type_not_same_as_int(void) { return {}; }
std::expected<int, NotSameAsInt> clval_error_type_not_same_as_int(void) { return {}; }
std::expected<int, NotSameAsInt> rval_error_type_not_same_as_int(void) { return {}; }
std::expected<int, NotSameAsInt> crval_error_type_not_same_as_int(void) { return {}; }

// clang-format off
void test() {
  // Test & overload
  {
    // U is not a specialization of std::expected
    {
      std::expected<void, int> f1;
      f1.and_then(lval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<int (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<U:error_type, E>
    {
      std::expected<void, int> f1;
      f1.and_then(lval_error_type_not_same_as_int);  // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<std::expected<int, NotSameAsInt> (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must have the same error_type as this expected}}
    }
  }

  // Test const& overload
  {
    // U is not a specialization of std::expected
    {
      const std::expected<void, int> f1;
      f1.and_then(clval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<int (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<U:error_type, E>
    {
      const std::expected<void, int> f1;
      f1.and_then(clval_error_type_not_same_as_int);  // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<std::expected<int, NotSameAsInt> (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must have the same error_type as this expected}}
    }
  }

  // Test && overload
  {
    // U is not a specialization of std::expected
    {
      std::expected<void, int> f1;
      std::move(f1).and_then(rval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<int (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<U:error_type, E>
    {
      std::expected<void, int> f1;
      std::move(f1).and_then(rval_error_type_not_same_as_int); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<std::expected<int, NotSameAsInt> (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must have the same error_type as this expected}}
    }
  }

  // Test const&& overload
  {
    // U is not a specialization of std::expected
    {
      const std::expected<void, int> f1;
      std::move(f1).and_then(crval_return_not_std_expected); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<int (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must be a specialization of std::expected}}
      // expected-error-re@*:* {{{{.*}}cannot be used prior to '::' because it has no members}}
      // expected-error-re@*:* {{no matching constructor for initialization of{{.*}}}}
    }

    // !std::is_same_v<U:error_type, E>
    {
      const std::expected<void, int> f1;
      std::move(f1).and_then(crval_error_type_not_same_as_int); // expected-note{{in instantiation of function template specialization 'std::expected<void, int>::and_then<std::expected<int, NotSameAsInt> (&)()>' requested here}}
      // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}The result of f() must have the same error_type as this expected}}
    }
  }
}
// clang-format on
