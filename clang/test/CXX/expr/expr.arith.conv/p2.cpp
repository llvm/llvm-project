// RUN: %clang_cc1 -verify %s -std=c++17 -Weverything -Wno-deprecated -Wno-float-equal
// RUN: %clang_cc1 -verify %s -std=c++2a -Wdeprecated

static enum E1 {} e1, e1b;
static enum E2 {} e2;
static double d;
extern void f();
extern bool b;

void f() {
  void(e1 * e1);
  void(e1 * e2); // expected-warning {{arithmetic between different enumeration types}}
  void(e1 * d); // expected-warning {{arithmetic between enumeration type 'enum E1' and floating-point type 'double'}}
  void(d * e1); // expected-warning {{arithmetic between floating-point type 'double' and enumeration type 'enum E1'}}

  void(e1 + e1);
  void(e1 + e2); // expected-warning {{arithmetic between different enumeration types}}
  void(e1 + d); // expected-warning {{arithmetic between enumeration type 'enum E1' and floating-point type 'double'}}
  void(d + e1); // expected-warning {{arithmetic between floating-point type 'double' and enumeration type 'enum E1'}}

#if __cplusplus > 201703L
  void(e1 <=> e1b); // expected-error {{include <compare>}}
  void(e1 <=> e2); // expected-error {{invalid operands}}
  void(e1 <=> d); // expected-error {{invalid operands}}
  void(d <=> e1); // expected-error {{invalid operands}}
#endif

  void(e1 < e1b);
  void(e1 < e2); // expected-warning {{comparison of different enumeration types}}
  void(e1 < d); // expected-warning {{comparison of enumeration type 'enum E1' with floating-point type 'double'}}
  void(d < e1); // expected-warning {{comparison of floating-point type 'double' with enumeration type 'enum E1'}}

  void(e1 == e1b);
  void(e1 == e2); // expected-warning {{comparison of different enumeration types}}
  void(e1 == d); // expected-warning {{comparison of enumeration type 'enum E1' with floating-point type 'double'}}
  void(d == e1); // expected-warning {{comparison of floating-point type 'double' with enumeration type 'enum E1'}}

  void(b ? e1 : e1b);
  void(b ? e1 : e2); // expected-warning {{conditional expression between different enumeration types}}
  void(b ? e1 : d); // expected-warning {{conditional expression between enumeration type 'enum E1' and floating-point type 'double'}}
  void(b ? d : e1); // expected-warning {{conditional expression between floating-point type 'double' and enumeration type 'enum E1'}}

  void(e1 = e1b);
  void(e1 = e2); // expected-error {{incompatible}}
  void(e1 = d); // expected-error {{incompatible}}
  void(d = e1); // FIXME: Should we warn on this?

  void(e1 += e1b); // expected-error {{incompatible}}
  void(e1 += e2); // expected-error {{incompatible}}
  void(e1 += d); // expected-error {{incompatible}}
  void(d += e1); // expected-warning {{compound assignment of floating-point type 'double' from enumeration type 'enum E1'}}
}
