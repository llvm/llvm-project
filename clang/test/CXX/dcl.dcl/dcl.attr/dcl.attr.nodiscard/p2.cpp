// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify=expected,cxx11,cxx11-17 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify=expected,cxx11-17,since-cxx17 -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify=expected,since-cxx17 -pedantic %s

struct [[nodiscard]] S {};
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
S get_s();
S& get_s_ref();

enum [[nodiscard]] E {};
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
E get_e();

[[nodiscard]] int get_i();
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
[[nodiscard]] volatile int &get_vi();
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}

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

[[nodiscard]] volatile char &(*fp)(); // expected-warning {{'nodiscard' attribute only applies to functions, classes, or enumerations}}
// cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
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
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}
struct LaterReason;
struct [[nodiscard("later reason")]] LaterReason {};
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}

ReasonStruct get_reason();
LaterReason get_later_reason();
[[nodiscard("another reason")]] int another_reason();
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}

[[nodiscard("conflicting reason")]] int conflicting_reason();
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}
[[nodiscard("special reason")]] int conflicting_reason();
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}

void cxx20_use() {
  get_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: reason}}
  get_later_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: later reason}}
  another_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: another reason}}
  conflicting_reason(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: special reason}}
}

namespace p1771 {
struct[[nodiscard("Don't throw me away!")]] ConvertTo{};
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}
struct S {
  [[nodiscard]] S();
  // cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
  [[nodiscard("Don't let that S-Char go!")]] S(char);
  // cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}
  S(int);
  [[gnu::warn_unused_result]] S(double);
  operator ConvertTo();
  [[nodiscard]] operator int();
  // cxx11-warning@-1 {{use of the 'nodiscard' attribute is a C++17 extension}}
  [[nodiscard("Don't throw away as a double")]] operator double();
  // cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}
};

struct[[nodiscard("Don't throw me away either!")]] Y{};
// cxx11-17-warning@-1 {{use of the 'nodiscard' attribute is a C++20 extension}}

void usage() {
  S();    // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  S('A'); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't let that S-Char go!}}
  S(1);
  S(2.2);
  Y(); // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away either!}}
  S s;
  ConvertTo{}; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}

  // AST is different in C++17 mode. Before, a move ctor for ConvertTo is there
  // as well, hence the constructor warning.

  // since-cxx17-warning@+2 {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}
  // cxx11-warning@+1 {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away!}}
  (ConvertTo) s;
  (int)s; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  (S)'c'; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't let that S-Char go!}}
  // since-cxx17-warning@+2 {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw me away!}}
  // cxx11-warning@+1 {{ignoring temporary created by a constructor declared with 'nodiscard' attribute: Don't throw me away!}}
  static_cast<ConvertTo>(s);
  static_cast<int>(s); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  static_cast<double>(s); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute: Don't throw away as a double}}
}
} // namespace p1771
