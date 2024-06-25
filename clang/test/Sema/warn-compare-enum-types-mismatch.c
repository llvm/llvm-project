// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wenum-compare -Wno-unused-comparison %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wenum-compare -Wno-unused-comparison %s

// In C enumerators (i.e enumeration constants) have type int (until C23). In
// order to support diagnostics such as -Wenum-compare we pretend they have the
// type of their enumeration.

typedef enum EnumA {
  A
} EnumA;

enum EnumB {
  B,
  B1 = 1,
  // In C++ this comparison doesnt warn as enumerators dont have the type of
  // their enumeration before the closing brace. We mantain the same behavior
  // in C.
  B2 = A == B1
};

enum {
  C
};

void foo(void) {
  enum EnumA a = A;
  enum EnumB b = B;
  A == B;
  // expected-warning@-1 {{comparison of different enumeration types}}
  a == (B);
  // expected-warning@-1 {{comparison of different enumeration types}}
  a == b;
  // expected-warning@-1 {{comparison of different enumeration types}}
  A > B;
  // expected-warning@-1 {{comparison of different enumeration types}}
  A >= b;
  // expected-warning@-1 {{comparison of different enumeration types}}
  a > b;
  // expected-warning@-1 {{comparison of different enumeration types}}
  (A) <= ((B));
  // expected-warning@-1 {{comparison of different enumeration types}}
  a < B;
  // expected-warning@-1 {{comparison of different enumeration types}}
  a < b;
  // expected-warning@-1 {{comparison of different enumeration types}}

  // In the following cases we purposefully differ from GCC and dont warn 
  a == C; 
  A < C;
  b >= C; 
}
