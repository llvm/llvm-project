// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -fsyntax-only -verify %s

using Callback = int (__attribute__((ms_abi)) *)(int);

// An ordinary lambda uses the target's default calling convention.
Callback bad = [](int x) { return x; }; // expected-error {{no viable conversion}} expected-note {{candidate function}}

// copy must change both the call operator and the function pointer conversion.
Callback good = [](int x) __attribute__((copy((Callback)nullptr))) { return x; };
Callback standard = [] [[gnu::copy((Callback)nullptr)]](int x) { return x; };
Callback generic = [](auto x) __attribute__((copy((Callback)nullptr))) { return x; };

int __attribute__((ms_abi)) function(int);
Callback from_function = [](int x) __attribute__((copy(function))) { return x; };

// Explicit, incompatible calling conventions are still diagnosed.
int __attribute__((sysv_abi, copy(function))) conflict(int); // expected-error {{ms_abi and cdecl attributes are not compatible}}

template <typename T> Callback make_callback() {
  return [](int x) __attribute__((copy((Callback)nullptr))) { return x; };
}
Callback instantiated = make_callback<int>();

template <typename T> T make_dependent_callback() {
  return [](int x) __attribute__((copy((T)nullptr))) { return x; };
}
Callback dependent = make_dependent_callback<Callback>();

template <int N> __attribute__((copy(function))) int templated(int x) { return x; }
Callback function_template = templated<0>;
