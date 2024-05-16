// RUN: %clang_cc1 %s -verify -fopenacc

// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel

// expected-error@+1{{OpenACC construct 'serial' cannot be used here; it can only be used in a statement context}}
#pragma acc serial

// expected-error@+1{{OpenACC construct 'kernels' cannot be used here; it can only be used in a statement context}}
#pragma acc kernels

// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel
int foo;
// expected-error@+1{{OpenACC construct 'serial' cannot be used here; it can only be used in a statement context}}
#pragma acc serial
int foo2;
// expected-error@+1{{OpenACC construct 'kernels' cannot be used here; it can only be used in a statement context}}
#pragma acc kernels
int foo3;

struct S {
// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel
int foo;
// expected-error@+1{{OpenACC construct 'serial' cannot be used here; it can only be used in a statement context}}
#pragma acc serial
int foo2;
// expected-error@+1{{OpenACC construct 'kernels' cannot be used here; it can only be used in a statement context}}
#pragma acc kernels
int foo3;
};

void func() {
  // FIXME: Should we disallow this on declarations, or consider this to be on
  // the initialization?
#pragma acc parallel
  int foo;

#pragma acc parallel
  {
#pragma acc parallel
    {
    }
  }

  {
// expected-error@+2{{expected statement}}
#pragma acc parallel
  }

  {
// expected-error@+2{{expected statement}}
#pragma acc serial
  }
  {
// expected-error@+2{{expected statement}}
#pragma acc kernels
  }

#pragma acc parallel
  while(0){}

#pragma acc parallel
  for(;;){}

#pragma acc parallel
#pragma acc parallel
  for(;;){}

// expected-error@+2{{expected statement}}
#pragma acc parallel
};
