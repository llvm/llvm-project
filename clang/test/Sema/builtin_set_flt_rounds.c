// RUN: %clang_cc1 -triple mipsel-unknown-linux -fsyntax-only %s -verify=expected,unsupported
// RUN: %clang_cc1 -triple x86_64-gnu-linux -fsyntax-only %s -verify
struct S {int a;};
void test_builtin_set_flt_rounds() {
  __builtin_set_flt_rounds(1); // unsupported-error {{builtin is not supported on this target}}
  struct S s;
  __builtin_set_flt_rounds(s); // expected-error {{passing 'struct S' to parameter of incompatible type}}
}
