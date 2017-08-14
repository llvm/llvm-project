// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify -Wc++17-extensions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -DEXT -Wc++17-extensions %s

struct [[nodiscard]] S {};
S get_s();
S& get_s_ref();

enum [[nodiscard]] E {};
E get_e();

[[nodiscard]] int get_i();

void f() {
  get_s(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_i(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_e(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Okay, warnings are not encouraged
  get_s_ref();
  (void)get_s();
  (void)get_i();
  (void)get_e();
}

#ifdef EXT
// expected-warning@4 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@8 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@11 {{use of the 'nodiscard' attribute is a C++17 extension}}
#endif
