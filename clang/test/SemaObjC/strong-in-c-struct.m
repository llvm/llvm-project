// RUN: %clang_cc1 -triple arm64-apple-ios11 -fobjc-arc -fblocks  -fobjc-runtime=ios-11.0 -fsyntax-only -verify %s

typedef struct {
  id a;
} Strong;

void callee_variadic(const char *, ...);

void test_variadic(void) {
  Strong t;
  callee_variadic("s", t); // expected-error {{cannot pass non-trivial C object of type 'Strong' by value to variadic function}}
}

void test_jump0(int cond) {
  switch (cond) {
  case 0:
    ;
    Strong x; // expected-note {{jump bypasses initialization of variable of non-trivial C struct type}}
    break;
  case 1: // expected-error {{cannot jump from switch statement to this case label}}
    x.a = 0;
    break;
  }
}

void test_jump1(void) {
  static void *ips[] = { &&L0 };
L0:  // expected-note {{possible target of indirect goto}}
  ;
  Strong x; // expected-note {{jump exits scope of variable with non-trivial destructor}}
  goto *ips; // expected-error {{cannot jump}}
}

typedef void (^BlockTy)(void);
void func(BlockTy);
void func2(Strong);

void test_block_scope0(int cond) {
  Strong x; // expected-note {{jump enters lifetime of block which captures a C struct that is non-trivial to destroy}}
  switch (cond) {
  case 0:
    func(^{ func2(x); });
    break;
  default: // expected-error {{cannot jump from switch statement to this case label}}
    break;
  }
}

void test_block_scope1(void) {
  static void *ips[] = { &&L0 };
L0:  // expected-note {{possible target of indirect goto}}
  ;
  Strong x; // expected-note {{jump exits scope of variable with non-trivial destructor}} expected-note {{jump exits lifetime of block which captures a C struct that is non-trivial to destroy}}
  func(^{ func2(x); });
  goto *ips; // expected-error {{cannot jump}}
}

void test_compound_literal0(int cond, id x) {
  switch (cond) {
  case 0:
    (void)(Strong){ .a = x }; // expected-note {{jump enters lifetime of a compound literal that is non-trivial to destruct}}
    break;
  default: // expected-error {{cannot jump from switch statement to this case label}}
    break;
  }
}

void test_compound_literal1(id x) {
  static void *ips[] = { &&L0 };
L0:  // expected-note {{possible target of indirect goto}}
  ;
  (void)(Strong){ .a = x }; // expected-note {{jump exits lifetime of a compound literal that is non-trivial to destruct}}
  goto *ips; // expected-error {{cannot jump}}
}
