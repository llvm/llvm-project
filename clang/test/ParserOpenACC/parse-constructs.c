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
#pragma acc enter data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc exit data clause list
  for(;;){}
  // expected-error@+3{{invalid OpenACC directive 'enter invalid'}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter invalid
  for(;;){}
  // expected-error@+3{{invalid OpenACC directive 'exit invalid'}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc exit invalid
  for(;;){}
  // expected-error@+2{{invalid OpenACC directive 'enter'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter
  for(;;){}
  // expected-error@+3{{invalid OpenACC directive 'exit }'}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc exit }
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

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel loop
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop clause list
  for(;;){}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc kernels loop clause list
  for(;;){}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc kernels loop
  for(;;){}

  int i = 0, j = 0, k = 0;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic
  i = j;
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic garbage
  i = j;
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic garbage clause list
  i = j;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic read
  i = j;
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic write clause list
  i = i + j;
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic update clause list
  i++;
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic capture clause list
  i = j++;


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
