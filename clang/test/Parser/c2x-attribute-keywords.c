// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,notc2x -Wno-strict-prototypes %s
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-none-linux-gnu -target-feature +sme -verify=expected,c2x %s

enum __arm_streaming E { // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
  One __arm_streaming, // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
  Two,
  Three __arm_streaming // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
};

enum __arm_streaming { Four }; // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
__arm_streaming enum E2 { Five }; // expected-error {{misplaced '__arm_streaming'}}

// FIXME: this diagnostic can be improved.
enum { __arm_streaming Six }; // expected-error {{expected identifier}}

// FIXME: this diagnostic can be improved.
enum E3 __arm_streaming { Seven }; // expected-error {{expected identifier or '('}}

struct __arm_streaming S1 { // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
  int i __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
  int __arm_streaming j; // expected-error {{'__arm_streaming' only applies to function types}}
  int k[10] __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
  int l __arm_streaming[10]; // expected-error {{'__arm_streaming' only applies to function types}}
  __arm_streaming int m, n; // expected-error {{'__arm_streaming' only applies to function types}}
  int o __arm_streaming : 12; // expected-error {{'__arm_streaming' only applies to function types}}
  int __arm_streaming : 0; // expected-error {{'__arm_streaming' only applies to function types}}
  int p, __arm_streaming : 0; // expected-error {{'__arm_streaming' cannot appear here}}
  int q, __arm_streaming r; // expected-error {{'__arm_streaming' cannot appear here}}
  __arm_streaming int; // expected-error {{'__arm_streaming' cannot appear here}} \
            // expected-warning {{declaration does not declare anything}}
};

__arm_streaming struct S2 { int a; }; // expected-error {{misplaced '__arm_streaming'}}
struct S3 __arm_streaming { int a; }; // expected-error {{'__arm_streaming' cannot appear here}} \
                                         expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}

union __arm_streaming U { // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
  double d __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types; type here is 'double'}}
  __arm_streaming int i; // expected-error {{'__arm_streaming' only applies to function types; type here is 'int'}}
};

__arm_streaming union U2 { double d; }; // expected-error {{misplaced '__arm_streaming'}}
union U3 __arm_streaming { double d; }; // expected-error {{'__arm_streaming' cannot appear here}} \
                                           expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}

struct __arm_streaming IncompleteStruct; // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
union __arm_streaming IncompleteUnion; // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}
enum __arm_streaming IncompleteEnum; // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}

__arm_streaming void f1(void); // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
void __arm_streaming f2(void); // expected-error {{'__arm_streaming' only applies to function types}}
void f3 __arm_streaming (void); // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
void f4(void) __arm_streaming;

void f5(int i __arm_streaming, __arm_streaming int j, int __arm_streaming k); // expected-error 3 {{'__arm_streaming' only applies to function types}}

void f6(a, b) __arm_streaming int a; int b; { // expected-error {{'__arm_streaming' cannot appear here}} \
                                                 c2x-warning {{deprecated}}
}

// FIXME: technically, an attribute list cannot appear here, but we currently
// parse it as part of the return type of the function, which is reasonable
// behavior given that we *don't* want to parse it as part of the K&R parameter
// declarations. It is disallowed to avoid a parsing ambiguity we already
// handle well.
int (*f7(a, b))(int, int) __arm_streaming int a; int b; { // c2x-warning {{deprecated}}
  return 0;
}

__arm_streaming int a, b; // expected-error {{'__arm_streaming' only applies to function types}}
int c __arm_streaming, d __arm_streaming; // expected-error 2 {{'__arm_streaming' only applies to function types}}

void f8(void) __arm_streaming {
  __arm_streaming int i, j; // expected-error {{'__arm_streaming' only applies to function types}}
  int k, l __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
}

__arm_streaming void f9(void) { // expected-error {{'__arm_streaming' cannot be applied to a declaration}}
  int i[10] __arm_streaming; // expected-error {{'__arm_streaming' only applies to function types}}
  int (*fp1)(void)__arm_streaming;
  int (*fp2 __arm_streaming)(void); // expected-error {{'__arm_streaming' cannot be applied to a declaration}}

  int * __arm_streaming *ipp; // expected-error {{'__arm_streaming' only applies to function types}}
}

void f10(int j[static 10] __arm_streaming, int k[*] __arm_streaming); // expected-error 2 {{'__arm_streaming' only applies to function types}}

void f11(void) {
  __arm_streaming {} // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming if (1) {} // expected-error {{'__arm_streaming' cannot be applied to a statement}}

  __arm_streaming switch (1) { // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming case 1: __arm_streaming break; // expected-error 2 {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming default: break; // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  }

  goto foo;
  __arm_streaming foo: (void)1; // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}}

  __arm_streaming for (;;); // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming while (1); // expected-error {{'__arm_streaming' cannot be applied to a statement}}
  __arm_streaming do __arm_streaming { } while(1); // expected-error 2 {{'__arm_streaming' cannot be applied to a statement}}

  __arm_streaming (void)1; // expected-error {{'__arm_streaming' cannot be applied to a statement}}

  __arm_streaming; // expected-error {{'__arm_streaming' cannot be applied to a statement}}

  (void)sizeof(int [4]__arm_streaming); // expected-error {{'__arm_streaming' only applies to function types}}
  (void)sizeof(struct __arm_streaming S3 { int a __arm_streaming; }); // expected-error {{'__arm_streaming' only applies to non-K&R-style functions}} \
                                                                      // expected-error {{'__arm_streaming' only applies to function types; type here is 'int'}}

  __arm_streaming return; // expected-error {{'__arm_streaming' cannot be applied to a statement}}

  __arm_streaming asm (""); // expected-error {{'__arm_streaming' cannot appear here}}
}

struct __arm_streaming S4 *s; // expected-error {{'__arm_streaming' cannot appear here}}
struct S5 {};
int c = sizeof(struct __arm_streaming S5); // expected-error {{'__arm_streaming' cannot appear here}}
