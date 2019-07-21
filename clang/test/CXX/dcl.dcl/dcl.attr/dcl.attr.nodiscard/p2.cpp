// RUN: %clang_cc1 -fsyntax-only -std=c++2a -verify -Wc++2a-extensions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify -Wc++17-extensions %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -DEXT -Wc++17-extensions -Wc++2a-extensions %s

struct [[nodiscard]] S {};
S get_s();
S& get_s_ref();

enum [[nodiscard]] E {};
E get_e();

[[nodiscard]] int get_i();
[[nodiscard]] volatile int &get_vi();

void f() {
  get_s(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_i(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_vi(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_e(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Okay, warnings are not encouraged
  get_s_ref();
  (void)get_s();
  (void)get_i();
  (void)get_vi();
  (void)get_e();
}

[[nodiscard]] volatile char &(*fp)();
void g() {
  fp(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // OK, warning suppressed.
  (void)fp();
}

namespace PR31526 {
typedef E (*fp1)();
typedef S (*fp2)();

typedef S S_alias;
typedef S_alias (*fp3)();

typedef fp2 fp2_alias;

void f() {
  fp1 one;
  fp2 two;
  fp3 three;
  fp2_alias four;

  one(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  two(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  three(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  four(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // These are all okay because of the explicit cast to void.
  (void)one();
  (void)two();
  (void)three();
  (void)four();
}
} // namespace PR31526

struct [[nodiscard("reason")]] ReasonStruct {};
struct LaterReason;
struct [[nodiscard("later reason")]] LaterReason {};

ReasonStruct get_reason();
LaterReason get_later_reason();
[[nodiscard("another reason")]] int another_reason();

[[nodiscard("conflicting reason")]] int conflicting_reason();
[[nodiscard("special reason")]] int conflicting_reason();

void cxx2a_use() {
  get_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: reason}}
  get_later_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: later reason}}
  another_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: another reason}}
  conflicting_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: special reason}}
}

#ifdef EXT
// expected-warning@5 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@9 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@12 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@13 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@29 {{use of the 'nodiscard' attribute is a C++17 extension}}
// expected-warning@65 {{use of the 'nodiscard' attribute is a C++2a extension}}
// expected-warning@67 {{use of the 'nodiscard' attribute is a C++2a extension}}
// expected-warning@71 {{use of the 'nodiscard' attribute is a C++2a extension}}
// expected-warning@73 {{use of the 'nodiscard' attribute is a C++2a extension}}
// expected-warning@74 {{use of the 'nodiscard' attribute is a C++2a extension}}
#endif
