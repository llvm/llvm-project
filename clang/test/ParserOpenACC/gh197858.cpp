// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
#pragma acc parallel
  using i; // expected-error{{using declaration requires a qualified name}}
#pragma acc loop
  using j; // expected-error{{using declaration requires a qualified name}}
#pragma acc parallel loop
  using k; // expected-error{{using declaration requires a qualified name}}
}
