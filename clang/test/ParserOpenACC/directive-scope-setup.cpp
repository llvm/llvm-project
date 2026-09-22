// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
#pragma acc parallel
  using i; // expected-error{{using declaration requires a qualified name}}
#pragma acc loop // expected-note{{'loop' construct is here}}
  using j; // expected-error{{using declaration requires a qualified name}} \
              expected-error{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
#pragma acc parallel loop // expected-note{{'parallel loop' construct is here}}
  using k; // expected-error{{using declaration requires a qualified name}} \
              expected-error{{OpenACC 'parallel loop' construct can only be applied to a 'for' loop}}
#pragma acc data default(none)
  using l; // expected-error{{using declaration requires a qualified name}}
}
