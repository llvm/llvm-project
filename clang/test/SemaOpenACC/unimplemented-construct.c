// RUN: %clang_cc1 %s -verify -fopenacc

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine

struct S {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
int foo;
};

void func() {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
  int foo;

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
  {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
    {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
    }
  }

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
  while(0){}

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
  for(;;){}

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine
};
