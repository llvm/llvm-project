// RUN: sed -e "s@ATTR_USE@__arm_streaming@g" -e "s@ATTR_NAME@__arm_streaming@g" %s > %t
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,notc2x -Wno-strict-prototypes %t
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,c2x %t
// RUN: sed -e "s@ATTR_USE@__arm_inout\(\"za\"\)@g" -e "s@ATTR_NAME@__arm_inout@g" %s > %t
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,notc2x -Wno-strict-prototypes %t
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,c2x %t

enum ATTR_USE E { // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  One ATTR_USE, // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  Two,
  Three ATTR_USE // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
};

enum ATTR_USE { Four }; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
ATTR_USE enum E2 { Five }; // expected-error {{misplaced 'ATTR_NAME'}}

// FIXME: this diagnostic can be improved.
enum { ATTR_USE Six }; // expected-error {{expected identifier}}

// FIXME: this diagnostic can be improved.
enum E3 ATTR_USE { Seven }; // expected-error {{expected identifier or '('}}

struct ATTR_USE S1 { // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  int i ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
  int ATTR_USE j; // expected-error {{'ATTR_NAME' only applies to function types}}
  int k[10] ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
  int l ATTR_USE[10]; // expected-error {{'ATTR_NAME' only applies to function types}}
  ATTR_USE int m, n; // expected-error {{'ATTR_NAME' only applies to function types}}
  int o ATTR_USE : 12; // expected-error {{'ATTR_NAME' only applies to function types}}
  int ATTR_USE : 0; // expected-error {{'ATTR_NAME' only applies to function types}}
  int p, ATTR_USE : 0; // expected-error {{'ATTR_NAME' cannot appear here}}
  int q, ATTR_USE r; // expected-error {{'ATTR_NAME' cannot appear here}}
  ATTR_USE int; // expected-error {{'ATTR_NAME' cannot appear here}} \
            // expected-warning {{declaration does not declare anything}}
};

ATTR_USE struct S2 { int a; }; // expected-error {{misplaced 'ATTR_NAME'}}
struct S3 ATTR_USE { int a; }; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                         expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

union ATTR_USE U { // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
  double d ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types; type here is 'double'}}
  ATTR_USE int i; // expected-error {{'ATTR_NAME' only applies to function types; type here is 'int'}}
};

ATTR_USE union U2 { double d; }; // expected-error {{misplaced 'ATTR_NAME'}}
union U3 ATTR_USE { double d; }; // expected-error {{'ATTR_NAME' cannot appear here}} \
                                           expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

struct ATTR_USE IncompleteStruct; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
union ATTR_USE IncompleteUnion; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}
enum ATTR_USE IncompleteEnum; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

ATTR_USE void f1(void); // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
void ATTR_USE f2(void); // expected-error {{'ATTR_NAME' only applies to function types}}
void f3 ATTR_USE (void); // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
void f4(void) ATTR_USE;

void f5(int i ATTR_USE, ATTR_USE int j, int ATTR_USE k); // expected-error 3 {{'ATTR_NAME' only applies to function types}}

void f6(a, b) ATTR_USE int a; int b; { // expected-error {{'ATTR_NAME' cannot appear here}} \
                                                 c2x-warning {{deprecated}}
}

// FIXME: technically, an attribute list cannot appear here, but we currently
// parse it as part of the return type of the function, which is reasonable
// behavior given that we *don't* want to parse it as part of the K&R parameter
// declarations. It is disallowed to avoid a parsing ambiguity we already
// handle well.
int (*f7(a, b))(int, int) ATTR_USE int a; int b; { // c2x-warning {{deprecated}}
  return 0;
}

ATTR_USE int a, b; // expected-error {{'ATTR_NAME' only applies to function types}}
int c ATTR_USE, d ATTR_USE; // expected-error 2 {{'ATTR_NAME' only applies to function types}}

void f8(void) ATTR_USE {
  ATTR_USE int i, j; // expected-error {{'ATTR_NAME' only applies to function types}}
  int k, l ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
}

ATTR_USE void f9(void) { // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}
  int i[10] ATTR_USE; // expected-error {{'ATTR_NAME' only applies to function types}}
  int (*fp1)(void)ATTR_USE;
  int (*fp2 ATTR_USE)(void); // expected-error {{'ATTR_NAME' cannot be applied to a declaration}}

  int * ATTR_USE *ipp; // expected-error {{'ATTR_NAME' only applies to function types}}
}

void f10(int j[static 10] ATTR_USE, int k[*] ATTR_USE); // expected-error 2 {{'ATTR_NAME' only applies to function types}}

void f11(void) {
  ATTR_USE {} // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE if (1) {} // expected-error {{'ATTR_NAME' cannot be applied to a statement}}

  ATTR_USE switch (1) { // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE case 1: ATTR_USE break; // expected-error 2 {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE default: break; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  }

  goto foo;
  ATTR_USE foo: (void)1; // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}}

  ATTR_USE for (;;); // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE while (1); // expected-error {{'ATTR_NAME' cannot be applied to a statement}}
  ATTR_USE do ATTR_USE { } while(1); // expected-error 2 {{'ATTR_NAME' cannot be applied to a statement}}

  ATTR_USE (void)1; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}

  ATTR_USE; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}

  (void)sizeof(int [4]ATTR_USE); // expected-error {{'ATTR_NAME' only applies to function types}}
  (void)sizeof(struct ATTR_USE S3 { int a ATTR_USE; }); // expected-error {{'ATTR_NAME' only applies to non-K&R-style functions}} \
                                                                      // expected-error {{'ATTR_NAME' only applies to function types; type here is 'int'}}

  ATTR_USE return; // expected-error {{'ATTR_NAME' cannot be applied to a statement}}

  ATTR_USE asm (""); // expected-error {{'ATTR_NAME' cannot appear here}}
}

struct ATTR_USE S4 *s; // expected-error {{'ATTR_NAME' cannot appear here}}
struct S5 {};
int c = sizeof(struct ATTR_USE S5); // expected-error {{'ATTR_NAME' cannot appear here}}

void invalid_parentheses1() __arm_inout; // expected-error {{expected '(' after ''__arm_inout''}}
void invalid_parentheses2() __arm_inout(; // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
void invalid_parentheses3() __arm_inout((); // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
void invalid_parentheses4() __arm_inout); // expected-error {{expected '(' after ''__arm_inout''}} \
                                          // expected-error {{expected function body after function declarator}}
void invalid_parentheses5() __arm_inout(());  // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
void invalid_parentheses6() __arm_inout("za"; // expected-error {{expected ')'}}
void invalid_parentheses7() __arm_streaming(; // expected-error {{expected parameter declarator}} \
                                              // expected-error {{expected ')'}} \
                                              // expected-note {{to match this '('}} \
                                              // expected-error {{function cannot return function type 'void ()'}} \
                                              // expected-error {{'__arm_streaming' only applies to function types; type here is 'int ()'}} \
                                              // expected-warning {{'__arm_streaming' only applies to non-K&R-style functions}}
void invalid_parentheses8() __arm_streaming();  // expected-error {{function cannot return function type 'void ()'}} \
                                                // expected-error {{'__arm_streaming' only applies to function types; type here is 'int ()'}} \
                                                // expected-warning {{'__arm_streaming' only applies to non-K&R-style functions}}
