// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=expected,pre26 -pedantic %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify=expected,compat -Wpre-c++26-compat %s
// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s

struct S {
  void a() = delete;
  void b() = delete(; // expected-error {{expected string literal}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  void c() = delete(); // expected-error {{expected string literal}}
  void d() = delete(42); // expected-error {{expected string literal}}
  void e() = delete("foo"[0]); // expected-error {{expected ')'}} expected-note {{to match this '('}} // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  void f() = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}

  S() = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  ~S() = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  S(const S&) = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  S(S&&) = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  S& operator=(const S&) = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
  S& operator=(S&&) = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
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
void e() = delete("foo"[0]); // expected-error {{expected ')'}} expected-note {{to match this '('}} // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}
void f() = delete("foo"); // pre26-warning {{'= delete' with a message is a C++2c extension}} compat-warning {{'= delete' with a message is incompatible with C++ standards before C++2c}}

constexpr const char *getMsg() { return "this is a message"; }
void func() = delete(getMsg()); // expected-error {{expected string literal}}

namespace CWG2876 {
using T = void ();
using U = int;

T a = delete ("hello"); // expected-error {{only functions can have deleted definitions}}
U b = delete ("hello"), c, d = delete ("hello"); // expected-error 2 {{only functions can have deleted definitions}}

struct C {
  T e = delete ("hello"); // expected-error {{'= delete' is a function definition and must occur in a standalone declaration}}
  U f = delete ("hello"); // expected-error {{cannot delete expression of type 'const char[6]'}}
};
}
