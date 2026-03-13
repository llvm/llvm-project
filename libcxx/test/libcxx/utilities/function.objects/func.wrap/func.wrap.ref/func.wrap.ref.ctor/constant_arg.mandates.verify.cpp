//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<auto f> constexpr function_ref(constant_arg_t<f>) noexcept;
// template<auto f, class U>
//   constexpr function_ref(constant_arg_t<f>, U&& obj) noexcept;
// template<auto f, class T>
//   constexpr function_ref(constant_arg_t<f>, cv T* obj) noexcept;

// Mandates: If is_pointer_v<F> || is_member_pointer_v<F> is true, then f != nullptr is true.

#include <functional>
#include <utility>

struct A {
  void f();
};

// clang-format off
void test() {
  std::function_ref<void()> f1(std::constant_arg<static_cast<void (*)()>(nullptr)>); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void(A)> f2(std::constant_arg<static_cast<void (A::*)()>(nullptr)>); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}

  int i;
  std::function_ref<void()> f3(std::constant_arg<static_cast<void (*)(int)>(nullptr)>, i); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}

  A a;
  std::function_ref<void()> f4(std::constant_arg<static_cast<void (A::*)()>(nullptr)>, a); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void()> f5(std::constant_arg<static_cast<void (*)(int*)>(nullptr)>, &i); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void()> f6(std::constant_arg<static_cast<void (A::*)()>(nullptr)>, &a); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'nullptr != nullptr': the function pointer should not be a nullptr}}
}
// clang-format on
