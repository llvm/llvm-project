// RUN: %clang_cc1 %s -verify -fopenacc

// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel

// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel
int foo;

struct S {
// expected-error@+1{{OpenACC construct 'parallel' cannot be used here; it can only be used in a statement context}}
#pragma acc parallel
int foo;

void mem_func() {
  // FIXME: Should we disallow this on declarations, or consider this to be on
  // the initialization?
#pragma acc parallel
  int foo;

#pragma acc parallel
  {
  }

#pragma acc parallel
  while(0){}

#pragma acc parallel
  for(;;){}

// expected-error@+2{{expected statement}}
#pragma acc parallel
}

};

template<typename T>
void func() {
  // FIXME: Should we disallow this on declarations, and consider this to be on
  // the initialization?
#pragma acc parallel
  int foo;

#pragma acc parallel
  {
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

