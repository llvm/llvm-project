// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,notc2x -Wno-strict-prototypes %s
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,c2x %s

enum __arm_inout("za") E { // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  One __arm_inout("za"), // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  Two,
  Three __arm_inout("za") // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
};

enum __arm_inout("za") { Four }; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
__arm_inout("za") enum E2 { Five }; // expected-error {{misplaced '__arm_inout'}}

// FIXME: this diagnostic can be improved.
enum { __arm_inout("za") Six }; // expected-error {{expected identifier}}

// FIXME: this diagnostic can be improved.
enum E3 __arm_inout("za") { Seven }; // expected-error {{expected identifier or '('}}

struct __arm_inout("za") S1 { // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  int i __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
  int __arm_inout("za") j; // expected-error {{'__arm_inout' only applies to function types}}
  int k[10] __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
  int l __arm_inout("za")[10]; // expected-error {{'__arm_inout' only applies to function types}}
  __arm_inout("za") int m, n; // expected-error {{'__arm_inout' only applies to function types}}
  int o __arm_inout("za") : 12; // expected-error {{'__arm_inout' only applies to function types}}
  int __arm_inout("za") : 0; // expected-error {{'__arm_inout' only applies to function types}}
  int p, __arm_inout("za") : 0; // expected-error {{'__arm_inout' cannot appear here}}
  int q, __arm_inout("za") r; // expected-error {{'__arm_inout' cannot appear here}}
  __arm_inout("za") int; // expected-error {{'__arm_inout' cannot appear here}} \
            // expected-warning {{declaration does not declare anything}}
};

__arm_inout("za") struct S2 { int a; }; // expected-error {{misplaced '__arm_inout'}}
struct S3 __arm_inout("za") { int a; }; // expected-error {{'__arm_inout' cannot appear here}} \
                                         expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

union __arm_inout("za") U { // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
  double d __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types; type here is 'double'}}
  __arm_inout("za") int i; // expected-error {{'__arm_inout' only applies to function types; type here is 'int'}}
};

__arm_inout("za") union U2 { double d; }; // expected-error {{misplaced '__arm_inout'}}
union U3 __arm_inout("za") { double d; }; // expected-error {{'__arm_inout' cannot appear here}} \
                                           expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

struct __arm_inout("za") IncompleteStruct; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
union __arm_inout("za") IncompleteUnion; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}
enum __arm_inout("za") IncompleteEnum; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

__arm_inout("za") void f1(void); // expected-error {{'__arm_inout' cannot be applied to a declaration}}
void __arm_inout("za") f2(void); // expected-error {{'__arm_inout' only applies to function types}}
void f3 __arm_inout("za") (void); // expected-error {{'__arm_inout' cannot be applied to a declaration}}
void f4(void) __arm_inout("za");

void f5(int i __arm_inout("za"), __arm_inout("za") int j, int __arm_inout("za") k); // expected-error 3 {{'__arm_inout' only applies to function types}}

void f6(a, b) __arm_inout("za") int a; int b; { // expected-error {{'__arm_inout' cannot appear here}} \
                                                 c2x-warning {{deprecated}}
}

// FIXME: technically, an attribute list cannot appear here, but we currently
// parse it as part of the return type of the function, which is reasonable
// behavior given that we *don't* want to parse it as part of the K&R parameter
// declarations. It is disallowed to avoid a parsing ambiguity we already
// handle well.
int (*f7(a, b))(int, int) __arm_inout("za") int a; int b; { // c2x-warning {{deprecated}}
  return 0;
}

__arm_inout("za") int a, b; // expected-error {{'__arm_inout' only applies to function types}}
int c __arm_inout("za"), d __arm_inout("za"); // expected-error 2 {{'__arm_inout' only applies to function types}}

void f8(void) __arm_inout("za") {
  __arm_inout("za") int i, j; // expected-error {{'__arm_inout' only applies to function types}}
  int k, l __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
}

__arm_inout("za") void f9(void) { // expected-error {{'__arm_inout' cannot be applied to a declaration}}
  int i[10] __arm_inout("za"); // expected-error {{'__arm_inout' only applies to function types}}
  int (*fp1)(void)__arm_inout("za");
  int (*fp2 __arm_inout("za"))(void); // expected-error {{'__arm_inout' cannot be applied to a declaration}}

  int * __arm_inout("za") *ipp; // expected-error {{'__arm_inout' only applies to function types}}
}

void f10(int j[static 10] __arm_inout("za"), int k[*] __arm_inout("za")); // expected-error 2 {{'__arm_inout' only applies to function types}}

void f11(void) {
  __arm_inout("za") {} // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") if (1) {} // expected-error {{'__arm_inout' cannot be applied to a statement}}

  __arm_inout("za") switch (1) { // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") case 1: __arm_inout("za") break; // expected-error 2 {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") default: break; // expected-error {{'__arm_inout' cannot be applied to a statement}}
  }

  goto foo;
  __arm_inout("za") foo: (void)1; // expected-error {{'__arm_inout' only applies to non-K&R-style functions}}

  __arm_inout("za") for (;;); // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") while (1); // expected-error {{'__arm_inout' cannot be applied to a statement}}
  __arm_inout("za") do __arm_inout("za") { } while(1); // expected-error 2 {{'__arm_inout' cannot be applied to a statement}}

  __arm_inout("za") (void)1; // expected-error {{'__arm_inout' cannot be applied to a statement}}

  __arm_inout("za"); // expected-error {{'__arm_inout' cannot be applied to a statement}}

  (void)sizeof(int [4]__arm_inout("za")); // expected-error {{'__arm_inout' only applies to function types}}
  (void)sizeof(struct __arm_inout("za") S3 { int a __arm_inout("za"); }); // expected-error {{'__arm_inout' only applies to non-K&R-style functions}} \
                                                                      // expected-error {{'__arm_inout' only applies to function types; type here is 'int'}}

  __arm_inout("za") return; // expected-error {{'__arm_inout' cannot be applied to a statement}}

  __arm_inout("za") asm (""); // expected-error {{'__arm_inout' cannot appear here}}
}

struct __arm_inout("za") S4 *s; // expected-error {{'__arm_inout' cannot appear here}}
struct S5 {};
int c = sizeof(struct __arm_inout("za") S5); // expected-error {{'__arm_inout' cannot appear here}}

void invalid_parentheses1() __arm_inout; // expected-error {{expected '(' after ''__arm_inout''}}
void invalid_parentheses2() __arm_inout(; // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
void invalid_parentheses3() __arm_inout((); // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
void invalid_parentheses4() __arm_inout); // expected-error {{expected '(' after ''__arm_inout''}} \
                                          // expected-error {{expected function body after function declarator}}
void invalid_parentheses5() __arm_inout(());  // expected-error {{expected string literal as argument of '__arm_inout' attribute}}
