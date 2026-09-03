// RUN: %clang_cc1 -fsyntax-only -verify %s

// Diagnostic coverage for the __builtin_ct_select Sema checks. Codegen
// behavior is tested separately in clang/test/CodeGen/builtin-ct-select.c.

struct S {
  int x;
};

// A well-formed call must not diagnose.
int test_valid(int c, int a, int b) {
  return __builtin_ct_select(c, a, b);
}

// The builtin requires exactly three arguments.
void test_too_few(int c, int a) {
  __builtin_ct_select(c, a); // expected-error {{too few arguments to function call, expected at least 3, have 2}}
}

void test_too_many(int c, int a, int b, int d) {
  __builtin_ct_select(c, a, b, d); // expected-error {{too many arguments to function call, expected 3, have 4}}
}

// The condition must be an integer type.
void test_noninteger_cond(struct S s, int a, int b) {
  __builtin_ct_select(s, a, b); // expected-error {{used type 'struct S' where arithmetic or pointer type is required}}
}

// The value operands must be scalar or vector types.
void test_nonscalar_operands(int c, struct S s) {
  __builtin_ct_select(c, s, s); // expected-error {{incompatible operand types ('struct S' and 'struct S')}}
}

// The two value operands must have the same type.
void test_mismatched_operands(int c, int a, int *p) {
  __builtin_ct_select(c, a, p); // expected-error {{incompatible operand types ('int' and 'int *')}}
}
