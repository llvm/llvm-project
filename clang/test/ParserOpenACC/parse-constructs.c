// RUN: %clang_cc1 %s -verify -fopenacc

void func() {

  // expected-error@+2{{expected OpenACC directive}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc
  for(;;){}

  // expected-error@+4{{expected OpenACC directive}}
  // expected-error@+3{{expected clause-list or newline in OpenACC directive}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc(whatever) routine

  // expected-error@+3{{expected OpenACC directive}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc) routine

  // expected-error@+2{{invalid OpenACC directive 'invalid'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc invalid
  for(;;){}

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel clause list
  for(;;){}
  // expected-error@+3{{expected clause-list or newline in OpenACC directive}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel() clause list
  for(;;){}
  // expected-error@+4{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel( clause list
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
  // expected-error@+3{{expected identifier}}
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

// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine
void routine_func();
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine clause list
void routine_func();

// expected-error@+2{{use of undeclared identifier 'func_name'}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (func_name)
// expected-error@+3{{use of undeclared identifier 'func_name'}}
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (func_name) clause list

// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (routine_func)
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (routine_func) clause list

// expected-error@+3{{expected ')'}}
// expected-note@+2{{to match this '('}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (routine_func())

// expected-error@+2{{expected identifier}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine()

// expected-error@+2{{expected identifier}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(int)
