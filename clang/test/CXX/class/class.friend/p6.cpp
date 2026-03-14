// RUN: %clang_cc1 -fsyntax-only -Wc++11-compat -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -Wc++11-compat -verify -std=c++11 %s

class A {
  friend static class B; // expected-error {{'static' is invalid in friend declarations}}
  friend extern class C; // expected-error {{'extern' is invalid in friend declarations}}
#if __cplusplus < 201103L
  friend register class E; // expected-error {{'register' is invalid in friend declarations}}
#else
  friend register class E; // expected-error {{'register' is invalid in friend declarations}}
#endif
  friend mutable class F; // expected-error {{'mutable' is invalid in friend declarations}}
  friend typedef class G; // expected-error {{'typedef' is invalid in friend declarations}}
  friend __thread class G; // expected-error {{'__thread' is invalid in friend declarations}}
  friend _Thread_local class G; // expected-error {{'_Thread_local' is invalid in friend declarations}}
  friend static _Thread_local class G; // expected-error {{'static _Thread_local' is invalid in friend declarations}}
#if __cplusplus < 201103L
  friend auto class D; // expected-warning {{incompatible with C++11}} expected-error {{'auto' is invalid in friend declarations}}
#else
  friend thread_local class G; // expected-error {{'thread_local' is invalid in friend declarations}}
#endif
};
