//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// template<auto c, class F>
//   constexpr function_ref(constant_wrapper<c, F>) noexcept;
// template<auto c, class F, class U>
//   constexpr function_ref(constant_wrapper<c, F>, U&& obj) noexcept;
// template<auto c, class F, class T>
//   constexpr function_ref(constant_wrapper<c, F>, cv T* obj) noexcept;

// Mandates: If is_pointer_v<F> || is_member_pointer_v<F> is true, then f != nullptr is true.

// For the first overload, f ArgTypes is not an empty pack and all types in remove_cvref_t<ArgTypes>...
// satisfy constexpr-param then constant_wrapper<INVOKE (f.value, remove_cvref_t<ArgTypes>::value...)>
// is not a valid type.

#include <functional>
#include <utility>

struct A {
  void f();
};

struct B {
  constexpr int operator()(std::constant_wrapper<42>) const { return 42; }
  constexpr int operator()(int) const { return 42; }
};

// clang-format off
void test() {
  std::function_ref<void()> f1(std::cw<static_cast<void (*)()>(nullptr)>); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void(A)> f2(std::cw<static_cast<void (A::*)()>(nullptr)>); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void(std::constant_wrapper<42>)> f33(std::cw<B{}>);
  // expected-error@*:* {{static assertion failed due to requirement '!requires { std::constant_wrapper<std::__cw_fixed_value<int>{42}, int>; }': cw(args...) should be equivalent to fn(args...), otherwise the intended behavior for a function_ref constructed from cw would be ambiguous}}

  int i;
  std::function_ref<void()> f3(std::cw<static_cast<void (*)(int)>(nullptr)>, i); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}

  A a;
  std::function_ref<void()> f4(std::cw<static_cast<void (A::*)()>(nullptr)>, a); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void()> f5(std::cw<static_cast<void (*)(int*)>(nullptr)>, &i); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}

  std::function_ref<void()> f6(std::cw<static_cast<void (A::*)()>(nullptr)>, &a); // expected-note-re{{in instantiation of function template specialization 'std::function_ref{{.*}}' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement '__f.value != nullptr': the function pointer should not be a nullptr}}
}
// clang-format on
