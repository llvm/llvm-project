// RUN: %clang_cc1 %s -verify -fopenacc

void func() {

  // expected-error@+1{{expected OpenACC directive}}
#pragma acc
  for(;;){}

  // expected-error@+2{{expected OpenACC directive}}
  // expected-error@+1{{invalid OpenACC clause 'whatever'}}
#pragma acc(whatever) routine

  // expected-error@+2{{expected OpenACC directive}}
  // expected-error@+1{{invalid OpenACC clause 'routine'}}
#pragma acc) routine

  // expected-error@+1{{invalid OpenACC directive 'invalid'}}
#pragma acc invalid
  for(;;){}

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
#pragma acc parallel clause list
  for(;;){}
  // expected-error@+2{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
#pragma acc parallel() clause list
  for(;;){}
  // expected-error@+3{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel( clause list
  for(;;){}
  // expected-error@+2{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
#pragma acc parallel() clause list
  for(;;){}
  // expected-error@+3{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel( clause list
  for(;;){}
  // expected-error@+3{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'serial' not yet implemented, pragma ignored}}
#pragma acc serial() clause list
  for(;;){}
  // expected-error@+4{{expected clause-list or newline in OpenACC directive}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'serial' not yet implemented, pragma ignored}}
#pragma acc serial( clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'serial' not yet implemented, pragma ignored}}
#pragma acc serial clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'kernels' not yet implemented, pragma ignored}}
#pragma acc kernels clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'data' not yet implemented, pragma ignored}}
#pragma acc data clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'exit data' not yet implemented, pragma ignored}}
#pragma acc exit data clause list
  for(;;){}
  // expected-error@+1{{invalid OpenACC directive 'enter invalid'}}
#pragma acc enter invalid
  for(;;){}
  // expected-error@+1{{invalid OpenACC directive 'exit invalid'}}
#pragma acc exit invalid
  for(;;){}
  // expected-error@+1{{invalid OpenACC directive 'enter'}}
#pragma acc enter
  for(;;){}
  // expected-error@+1{{expected identifier}}
#pragma acc exit }
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'host_data' not yet implemented, pragma ignored}}
#pragma acc host_data clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop clause list
  for(;;){}
  // expected-error@+1{{invalid OpenACC clause 'invalid'}}
#pragma acc parallel invalid clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'invalid'}}
  // expected-warning@+1{{OpenACC construct 'serial' not yet implemented, pragma ignored}}
#pragma acc serial invalid clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'parallel loop' not yet implemented, pragma ignored}}
#pragma acc parallel loop clause list
  for(;;){}

  // expected-warning@+1{{OpenACC construct 'parallel loop' not yet implemented, pragma ignored}}
#pragma acc parallel loop
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop clause list
  for(;;){}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'kernels loop' not yet implemented, pragma ignored}}
#pragma acc kernels loop clause list
  for(;;){}
  // expected-warning@+1{{OpenACC construct 'kernels loop' not yet implemented, pragma ignored}}
#pragma acc kernels loop
  for(;;){}

  int i = 0, j = 0, k = 0;
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic
  i = j;
  // expected-error@+2{{invalid OpenACC clause 'garbage'}}
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic garbage
  i = j;
  // expected-error@+2{{invalid OpenACC clause 'garbage'}}
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic garbage clause list
  i = j;
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic read
  i = j;
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic write clause list
  i = i + j;
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic update clause list
  i++;
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'atomic' not yet implemented, pragma ignored}}
#pragma acc atomic capture clause list
  i = j++;


  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'declare' not yet implemented, pragma ignored}}
#pragma acc declare clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'shutdown' not yet implemented, pragma ignored}}
#pragma acc shutdown clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set clause list
  for(;;){}
  // expected-error@+2{{invalid OpenACC clause 'clause'}}
  // expected-warning@+1{{OpenACC construct 'update' not yet implemented, pragma ignored}}
#pragma acc update clause list
  for(;;){}
}

// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine
void routine_func();
// expected-error@+2{{invalid OpenACC clause 'clause'}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine clause list
void routine_func();

// expected-error@+2{{use of undeclared identifier 'func_name'}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine (func_name)
// expected-error@+3{{use of undeclared identifier 'func_name'}}
// expected-error@+2{{invalid OpenACC clause 'clause'}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine (func_name) clause list

// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine (routine_func)
// expected-error@+2{{invalid OpenACC clause 'clause'}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine (routine_func) clause list

// expected-error@+3{{expected ')'}}
// expected-note@+2{{to match this '('}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine (routine_func())

// expected-error@+2{{expected identifier}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine()

// expected-error@+2{{expected identifier}}
// expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine(int)
