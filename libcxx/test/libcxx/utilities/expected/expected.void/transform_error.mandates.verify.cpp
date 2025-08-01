//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// With clang-cl, some warnings have a 'which is a Microsoft extension' suffix
// which break the tests. But #102851 will turn it into an error, making the test pass.
// However, upstream libcxx buildbots do not build clang from source while testing, so
// this tests still expected to fail on these bots.
//
// TODO(LLVM 22): Remove '0-1' from 'expected-error-re@*:* 0-1 {{union member {{.*}} has reference type {{.*}}}}'
// and remove 'expected-warning-re@*:* 0-1 {{union member {{.*}} has reference type {{.*}}, which is a Microsoft extension}}'
// and remove 'expected-error-re@*:* 0-1 {{call to deleted constructor of {{.*}}}}'
// once LLVM 22 releases. See See https://github.com/llvm/llvm-project/issues/104885.

// Test the mandates

// template<class F> constexpr auto transform_error(F&& f) &;
// template<class F> constexpr auto transform_error(F&& f) const &;
//
// Let G be remove_cv_t<invoke_result_t<F, decltype(error())>>
// G is a valid template argument for unexpected ([expected.un.general]) and the declaration
// G g(invoke(std::forward<F>(f), error())); is well-formed.

// template<class F> constexpr auto transform_error(F&& f) &&;
// template<class F> constexpr auto transform_error(F&& f) const &&;
//
// Let G be remove_cv_t<invoke_result_t<F, decltype(std::move(error()))>>.
// G is a valid template argument for unexpected ([expected.un.general]) and the declaration
// G g(invoke(std::forward<F>(f), std::move(error()))); is well-formed.

#include <expected>
#include <utility>

static int val;

template <class T>
std::unexpected<int> return_unexpected(T) {
  return std::unexpected<int>(1);
}

template <class T>
int& return_no_object(T) {
  return val;
}

// clang-format off
void test() {

  // Test & overload
  {
    std::expected<void, int> e;
    e.transform_error(return_unexpected<int&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    // expected-error-re@*:* {{static assertion failed {{.*}}A program that instantiates expected<T, E> with a E that is not a valid argument for unexpected<E> is ill-formed}}
    // expected-error-re@*:* 0-1 {{call to deleted constructor of {{.*}}}}
    // expected-error-re@*:* 0-1 {{union member {{.*}} has reference type {{.*}}}}

    e.transform_error(return_no_object<int&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    // expected-error-re@*:* {{static assertion failed {{.*}}A program that instantiates expected<T, E> with a E that is not a valid argument for unexpected<E> is ill-formed}}
    // expected-warning-re@*:* 0-1 {{union member {{.*}} has reference type {{.*}}, which is a Microsoft extension}}
  }

  // Test const& overload
  {
    const std::expected<void, int> e;
    e.transform_error(return_unexpected<const int &>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    e.transform_error(return_no_object<const int &>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    // expected-error-re@*:* 0-1 {{call to deleted constructor of {{.*}}}}
  }

  // Test && overload
  {
    std::expected<void, int> e;
    std::move(e).transform_error(return_unexpected<int&&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    std::move(e).transform_error(return_no_object<int&&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
  }

  // Test const&& overload
  {
    const std::expected<void, int> e;
    std::move(e).transform_error(return_unexpected<const int&&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
    std::move(e).transform_error(return_no_object<const int&&>); // expected-error-re@*:* {{static assertion failed {{.*}}The result of {{.*}} must be a valid template argument for unexpected}}
    // expected-error-re@*:* 0-1 {{no matching constructor for initialization of{{.*}}}}
  }
}
// clang-format on
