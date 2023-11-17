// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
  // expected-error@+2{{invalid OpenACC directive 'invalid'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc invalid
  for(;;){}

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel() clause list
}
