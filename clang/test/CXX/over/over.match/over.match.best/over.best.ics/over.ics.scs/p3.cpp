// RUN: %clang_cc1 -std=c++1z -verify %s

void f(void() noexcept); // expected-note {{no known conversion from 'void ()' to 'void (*)() noexcept'}}
void f(void()) = delete; // expected-note {{explicitly deleted}}

void g();
void h() noexcept;

void test() {
  f(g); // expected-error {{call to deleted}}
  f(h);
}
