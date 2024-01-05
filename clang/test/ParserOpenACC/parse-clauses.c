// RUN: %clang_cc1 %s -verify -fopenacc -std=c99
// RUNX: %clang_cc1 %s -verify -fopenacc
// RUNX: %clang_cc1 %s -verify -fopenacc -x c++

void func() {

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data finalize

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data finalize finalize

  // expected-error@+2{{invalid OpenACC clause 'invalid'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data finalize invalid

  // expected-error@+2{{invalid OpenACC clause 'invalid'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data finalize invalid invalid finalize

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc enter data seq finalize

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc host_data if_present

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc host_data if_present, if_present

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop seq independent auto

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop seq, independent auto

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop seq independent, auto

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc kernels loop seq independent auto

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop seq, independent auto

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc parallel loop seq independent, auto


  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop , seq

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc loop seq,

}

void DefaultClause() {
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop default
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default, seq
  for(;;){}

  // expected-error@+4{{expected identifier}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(
  for(;;){}

  // expected-error@+4{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default( seq
  for(;;){}

  // expected-error@+4{{expected identifier}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(, seq
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default)
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default) seq
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default), seq
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default()
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default() seq
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(), seq
  for(;;){}

  // expected-error@+2{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(invalid)
  for(;;){}

  // expected-error@+2{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(auto) seq
  for(;;){}

  // expected-error@+2{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(invalid), seq
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(none)
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial default(present), seq
  for(;;){}


}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine worker, vector, seq, nohost
void bar();

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(bar) worker, vector, seq, nohost
