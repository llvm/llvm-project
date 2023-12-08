// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify %s

// Check that we emit the correct warnings in various situations where the C++11
// spelling of the `address_space` attribute is applied to a declaration instead
// of a type. Also check that the attribute can instead be applied to the type.

void f([[clang::address_space(1)]] int* param) { // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  [[clang::address_space(1)]] int* local1; // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  int* local2 [[clang::address_space(1)]]; // expected-error {{automatic variable qualified with an address space}} expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  int [[clang::address_space(1)]] * local3;
  int* [[clang::address_space(1)]] local4; // expected-error {{automatic variable qualified with an address space}}

  for ([[clang::address_space(1)]] int* p = nullptr; p; ++p) {} // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  for (; [[clang::address_space(1)]] int* p = nullptr; ) {} // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  while([[clang::address_space(1)]] int* p = nullptr) {} // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  if ([[clang::address_space(1)]] int* p = nullptr) {} // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  try {
  } catch([[clang::address_space(1)]] int& i) { // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  }

  for (int [[clang::address_space(1)]] * p = nullptr; p; ++p) {}
  for (; int [[clang::address_space(1)]] * p = nullptr; ) {}
  while(int [[clang::address_space(1)]] * p = nullptr) {}
  if (int [[clang::address_space(1)]] * p = nullptr) {}
  try {
  } catch(int [[clang::address_space(1)]] & i) {
  }
}

[[clang::address_space(1)]] int* return_value(); // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
int [[clang::address_space(1)]] * return_value();

[[clang::address_space(1)]] int global1; // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
int global2 [[clang::address_space(1)]]; // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
int [[clang::address_space(1)]] global3;
int [[clang::address_space(1)]] global4;

struct [[clang::address_space(1)]] S { // expected-error {{'address_space' attribute cannot be applied to a declaration}}
  [[clang::address_space(1)]] int* member_function_1(); // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
  int [[clang::address_space(1)]] * member_function_2();
};

template <class T>
[[clang::address_space(1)]] T var_template_1; // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
template <class T>
T [[clang::address_space(1)]] var_template_2;

using void_ptr [[clang::address_space(1)]] = void *; // expected-warning {{applying attribute 'address_space' to a declaration is deprecated; apply it to the type instead}}
// Intentionally using the same alias name to check that the aliases define the
// same type.
using void_ptr = void [[clang::address_space(1)]] *;

namespace N {}
[[clang::address_space(1)]] using namespace N; // expected-error {{'address_space' attribute cannot be applied to a declaration}}
