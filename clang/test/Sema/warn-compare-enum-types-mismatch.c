// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wenum-compare -Wno-unused-comparison %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wenum-compare -Wno-unused-comparison %s

typedef enum EnumA {
  A
} EnumA;

enum EnumB {
  B
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
