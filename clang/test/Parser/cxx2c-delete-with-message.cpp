// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

struct S {
  void a() = delete;
  void b() = delete(; // expected-error {{expected string literal}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  void c() = delete(); // expected-error {{expected string literal}}
  void d() = delete(42); // expected-error {{expected string literal}}
  void e() = delete("foo"[0]); // expected-error {{expected ')'}} expected-note {{to match this '('}}
  void f() = delete("foo");

  S() = delete("foo");
  ~S() = delete("foo");
  S(const S&) = delete("foo");
  S(S&&) = delete("foo");
  S& operator=(const S&) = delete("foo");
  S& operator=(S&&) = delete("foo");
};

struct T {
  T() = delete(); // expected-error {{expected string literal}}
  ~T() = delete(); // expected-error {{expected string literal}}
  T(const T&) = delete(); // expected-error {{expected string literal}}
  T(T&&) = delete(); // expected-error {{expected string literal}}
  T& operator=(const T&) = delete(); // expected-error {{expected string literal}}
  T& operator=(T&&) = delete(); // expected-error {{expected string literal}}
};

void a() = delete;
void b() = delete(; // expected-error {{expected string literal}} expected-error {{expected ')'}} expected-note {{to match this '('}}
void c() = delete(); // expected-error {{expected string literal}}
void d() = delete(42); // expected-error {{expected string literal}}
void e() = delete("foo"[0]); // expected-error {{expected ')'}} expected-note {{to match this '('}}
void f() = delete("foo");
