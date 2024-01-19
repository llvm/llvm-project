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

void IfClause() {
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop if
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if, seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(
  for(;;){}

  // expected-error@+2{{use of undeclared identifier 'seq'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if( seq
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{use of undeclared identifier 'seq'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(, seq
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if)
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if) seq
  for(;;){}

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if), seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if()
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if() seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(), seq
  for(;;){}

  // expected-error@+2{{use of undeclared identifier 'invalid_expr'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(invalid_expr)
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if() seq
  for(;;){}

  int i, j;

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(i > j)
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial if(1+5>3), seq
  for(;;){}
}

void SyncClause() {
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self, seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self(
  for(;;){}

  // expected-error@+2{{use of undeclared identifier 'seq'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self( seq
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{use of undeclared identifier 'seq'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self(, seq
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self)
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self) seq
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self), seq
  for(;;){}


  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self(), seq
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self(,), seq
  for(;;){}

  // expected-error@+2{{use of undeclared identifier 'invalid_expr'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial loop self(invalid_expr), seq
  for(;;){}

  int i, j;

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial self(i > j
  for(;;){}

  // expected-error@+2{{use of undeclared identifier 'seq'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial self(i > j, seq
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial self(i > j)
  for(;;){}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial self(1+5>3), seq
  for(;;){}
}

struct Members {
  int value;
  char array[5];
};
struct HasMembersArray {
  struct Members MemArr[4];
};

void VarListClauses() {
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy, seq

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy)

  // expected-error@+3{{expected '('}}
  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy), seq

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(, seq

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy()

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(), seq

  struct Members s;
  struct HasMembersArray HasMem;

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(s.array[s.value]), seq

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(s.array[s.value], s.array[s.value :5] ), seq

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[3].array[1]), seq

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[3].array[1:4]), seq

  // expected-error@+2{{OpenMP array section is not allowed here}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1]), seq

  // expected-error@+2{{OpenMP array section is not allowed here}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1:2]), seq

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[:]), seq

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[::]), seq

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ']'}}
  // expected-note@+2{{to match this '['}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[: :]), seq

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial copy(HasMem.MemArr[3:]), seq

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial use_device(s.array[s.value] s.array[s.value :5] ), seq

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc serial use_device(s.array[s.value : 5]), seq
}

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine worker, vector, seq, nohost
void bar();

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(bar) worker, vector, seq, nohost
