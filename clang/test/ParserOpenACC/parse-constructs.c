// RUN: %clang_cc1 %s -verify -fopenacc

void func() {

  // expected-error@+2{{expected OpenACC directive}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc
  for(;;){}

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
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc kernels clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc host_data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel invalid clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel loop clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc kernels loop clause list
  for(;;){}

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc declare clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc init clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc shutdown clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc set clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc update clause list
  for(;;){}
}
