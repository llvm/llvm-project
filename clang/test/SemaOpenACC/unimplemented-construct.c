// RUN: %clang_cc1 %s -verify -fopenacc

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq

struct S {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
int foo;
};

void func() {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
  int foo;

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
  {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
    {
// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
    }
  }

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
  while(0){}

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
  for(;;){}

// expected-warning@+1{{OpenACC construct 'routine' with implicit function not yet implemented, pragma ignored}}
#pragma acc routine seq
};
