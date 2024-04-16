// RUN: %clang_cc1 -verify -std=c++23 -Wpre-c++23-compat %s

constexpr int h(int n) {
  if (!n)
    return 0;
  static const int m = n; // expected-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++23}}
  return m;
}

constexpr int i(int n) {
  if (!n)
    return 0;
  thread_local const int m = n; // expected-warning {{definition of a thread_local variable in a constexpr function is incompatible with C++ standards before C++23}}
  return m;
}

constexpr int g() {
  goto test; // expected-warning {{use of this statement in a constexpr function is incompatible with C++ standards before C++23}}
test:
  return 0;
}

constexpr void h() {
label:; // expected-warning {{use of this statement in a constexpr function is incompatible with C++ standards before C++23}}
}

struct NonLiteral { // expected-note 2 {{'NonLiteral' is not literal}}
  NonLiteral() {}
};

constexpr void non_literal() {
  NonLiteral n; // expected-warning {{definition of a variable of non-literal type in a constexpr function is incompatible with C++ standards before C++23}}
}

constexpr void non_literal2(bool b) {
  if (!b)
    NonLiteral n; // expected-warning {{definition of a variable of non-literal type in a constexpr function is incompatible with C++ standards before C++23}}
}

constexpr int c_thread_local(int n) {
  if (!n)
    return 0;
  static _Thread_local int a; // expected-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++23}}
  _Thread_local int b;        // // expected-error {{'_Thread_local' variables must have global storage}}
  return 0;
}

constexpr int gnu_thread_local(int n) {
  if (!n)
    return 0;
  static __thread int a; // expected-warning {{definition of a static variable in a constexpr function is incompatible with C++ standards before C++23}}
  __thread int b;        // expected-error {{'__thread' variables must have global storage}}
  return 0;
}
