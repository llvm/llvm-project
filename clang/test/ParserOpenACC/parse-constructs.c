// RUN: %clang_cc1 %s -verify -fopenacc



void func() {

  //expected-error@+1{{invalid OpenACC directive 'invalid'}}
#pragma acc invalid
  for(;;){}

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel clause list
  for(;;){}
  // expected-error@+3{{expected clause-list or newline in pragma directive}}
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
  // expected-error@+1{{invalid OpenACC directive 'enter invalid'}}
#pragma acc enter invalid clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc exit data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc host_data clause list
  for(;;){}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop clause list
  for(;;){}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc cache
  for(;;){}
  // expected-warning@+2{{OpenACC 'cache' 'var-list' parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc cache(var,list)
  for(;;){}
  // expected-warning@+2{{OpenACC 'cache' 'var-list' parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc cache(readonly:var,list)
  for(;;){}
  // expected-warning@+3{{OpenACC 'cache' 'var-list' parsing not yet implemented}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc cache(readonly:var,list) invalid clause list
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

  int i = 0, j = 0, k = 0;
  // expected-error@+1{{invalid OpenACC 'atomic-clause' 'garbage'; expected 'read', 'write', 'update', or 'capture'}}
#pragma acc atomic garbage
  i = j;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic read
  i = j;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic write
  i = i + j;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic update
  i++;
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic capture
  i = j++;

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc atomic capture invalid clause list
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

// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine clause list
void routine_func(void ) {}

  // expected-warning@+3{{OpenACC 'routine-name' parsing not yet implemented}}
// expected-warning@+2{{OpenACC clause parsing not yet implemented}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(routine_func) clause list

void func2() {
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc wait clause-list
  for(;;){}
  // expected-warning@+3{{OpenACC 'wait-argument' parsing not yet implemented}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc wait(1::) clause-list
  for(;;){}
}
