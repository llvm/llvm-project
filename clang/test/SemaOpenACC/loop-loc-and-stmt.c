// RUN: %clang_cc1 %s -verify -fopenacc

// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop

// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop
int foo;

struct S {
// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop
  int i;
};

void func() {
  // expected-error@+2{{expected expression}}
#pragma acc loop
  int foo;

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  while(0);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  do{}while(0);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  {}

#pragma acc loop
  for(int i = 0; i < 5; ++i);
}
