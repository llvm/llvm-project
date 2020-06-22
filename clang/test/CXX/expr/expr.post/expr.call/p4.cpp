// RUN: %clang_cc1 -verify %s

void a(int x = 0, int y); // #1 expected-error {{missing default argument on parameter 'y'}}
void b() {
  a(); // expected-error {{no matching function}} expected-note@#1 {{requires 2 arguments, but 0 were provided}}
  a(0); // expected-error {{no matching function}} expected-note@#1 {{requires 2 arguments, but 1 was provided}}
  a(0, 0);
}

void a(int x, int y = 0);
void c() {
  a();
  a(0);
  a(0, 0);
}

template<typename ...T> void f(int x = 0, T ...); // #2
void g() {
  f<int>(); // expected-error {{no matching function}} expected-note@#2 {{requires 2 arguments, but 0 were provided}}
  f<int>(0); // expected-error {{no matching function}} expected-note@#2 {{requires 2 arguments, but 1 was provided}}
  f<int>(0, 0);
}
