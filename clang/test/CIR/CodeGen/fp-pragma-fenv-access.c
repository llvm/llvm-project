// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -verify -emit-cir -o -

extern void use(float f);
float a, b;

// expected-error@+1 {{ClangIR code gen Not Yet Implemented: STDC FENV_ACCESS}}
void fenv_access(void) {
  #pragma STDC FENV_ACCESS ON
  __builtin_set_flt_rounds(0);
  use(a + b);
}
