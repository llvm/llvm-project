// RUN: %clang_cc1 %s -verify -fopenacc

// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine

struct S {
// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare 
int foo;
};

void func() {
// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
  int foo;

// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
  {
// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
    {
// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
    }
  }

// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
  while(0){}

// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
  for(;;){}

// expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare
};

