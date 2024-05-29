// RUN: %clang_cc1 %s -verify -fopenacc -std=c99
// RUNX: %clang_cc1 %s -verify -fopenacc
// RUNX: %clang_cc1 %s -verify -fopenacc -x c++

void func() {

  // expected-warning@+2{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data finalize

  // expected-warning@+3{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data finalize finalize

  // expected-warning@+3{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-error@+2{{invalid OpenACC clause 'invalid'}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data finalize invalid

  // expected-warning@+3{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-error@+2{{invalid OpenACC clause 'invalid'}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data finalize invalid invalid finalize

  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'finalize' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'enter data' not yet implemented, pragma ignored}}
#pragma acc enter data seq finalize

  // expected-warning@+2{{OpenACC clause 'if_present' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'host_data' not yet implemented, pragma ignored}}
#pragma acc host_data if_present

  // expected-warning@+3{{OpenACC clause 'if_present' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'if_present' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'host_data' not yet implemented, pragma ignored}}
#pragma acc host_data if_present, if_present

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop seq independent auto

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop seq, independent auto

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop seq independent, auto

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'kernels loop' not yet implemented, pragma ignored}}
#pragma acc kernels loop seq independent auto
  for(;;){}

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop seq, independent auto
  {}

  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'independent' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'auto' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'parallel loop' not yet implemented, pragma ignored}}
#pragma acc parallel loop seq independent, auto
  {}


  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop , seq

  // expected-error@+3{{expected identifier}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop seq,

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse()
  for(;;){}

  // expected-error@+3{{invalid tag 'unknown' on 'collapse' clause}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(unknown:)
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(force:)
  for(;;){}

  // expected-error@+3{{invalid tag 'unknown' on 'collapse' clause}}
  // expected-warning@+2{{OpenACC clause 'collapse' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(unknown:5)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'collapse' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(force:5)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'collapse' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(5)
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop collapse(5, 6)
  for(;;){}
}

void DefaultClause() {
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop default
  for(;;){}

  // expected-error@+1{{expected '('}}
#pragma acc serial default seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default, seq
  for(;;){}

  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default(
  for(;;){}

  // expected-error@+3{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default( seq
  for(;;){}

  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default(, seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial default)
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial default), seq
  for(;;){}

  // expected-error@+1{{expected identifier}}
#pragma acc serial default()
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default() seq
  for(;;){}

  // expected-error@+2{{expected identifier}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default(), seq
  for(;;){}

  // expected-error@+1{{invalid value for 'default' clause; expected 'present' or 'none'}}
#pragma acc serial default(invalid)
  for(;;){}

  // expected-error@+2{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default(auto) seq
  for(;;){}

  // expected-error@+2{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default(invalid), seq
  for(;;){}

#pragma acc serial default(none)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial default(present), seq
  for(;;){}
}

void IfClause() {
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop if
  for(;;){}

  // expected-error@+1{{expected '('}}
#pragma acc serial if seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial if, seq
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if(
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if( seq
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if(, seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if)
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if) seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if), seq
  for(;;){}

  // expected-error@+1{{expected expression}}
#pragma acc serial if()
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial if() seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial if(), seq
  for(;;){}

  // expected-error@+1{{use of undeclared identifier 'invalid_expr'}}
#pragma acc serial if(invalid_expr)
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial if() seq
  for(;;){}

  int i, j;

#pragma acc serial if(i > j)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial if(1+5>3), seq
  for(;;){}
}

void SelfClause() {
  // expected-warning@+2{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self
  for(;;){}

  // expected-warning@+3{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self, seq
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self(
  for(;;){}

  // expected-error@+4{{use of undeclared identifier 'seq'}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self( seq
  for(;;){}

  // expected-error@+5{{expected expression}}
  // expected-error@+4{{use of undeclared identifier 'seq'}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self(, seq
  for(;;){}

  // expected-error@+3{{expected identifier}}
  // expected-warning@+2{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self)
  for(;;){}

  // expected-error@+3{{expected identifier}}
  // expected-warning@+2{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self) seq
  for(;;){}

  // expected-error@+3{{expected identifier}}
  // expected-warning@+2{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self), seq
  for(;;){}


  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self(), seq
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self(,), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'invalid_expr'}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'serial loop' not yet implemented, pragma ignored}}
#pragma acc serial loop self(invalid_expr), seq
  for(;;){}

  int i, j;

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial self(i > j
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial self(i > j, seq
  for(;;){}

  // expected-warning@+1{{left operand of comma operator has no effect}}
#pragma acc serial self(i, j)
  for(;;){}

#pragma acc serial self(i > j)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
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

// On 'update', self behaves differently and requires parens, plus takes a var-list instead.
void SelfUpdate() {
  struct Members s;

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'update' not yet implemented, pragma ignored}}
#pragma acc update self
  for(;;){}

  // expected-error@+6{{use of undeclared identifier 'zero'}}
  // expected-error@+5{{expected ','}}
  // expected-error@+4{{expected expression}}
  // expected-warning@+3{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'update' not yet implemented, pragma ignored}}
#pragma acc update self(zero : s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+3{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'update' not yet implemented, pragma ignored}}
#pragma acc update self(s.array[s.value : 5], s.value), seq
  for(;;){}
}

void VarListClauses() {
  // expected-error@+1{{expected '('}}
#pragma acc serial copy
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy, seq
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial copy)
  for(;;){}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial copy), seq
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial copy(
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial copy(, seq
  for(;;){}

  // expected-error@+1{{expected expression}}
#pragma acc serial copy()
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(), seq
  for(;;){}

  struct Members s;
  struct HasMembersArray HasMem;

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(s.array[s.value]), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(s.array[s.value], s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[3].array[1]), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[3].array[1:4]), seq
  for(;;){}

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1]), seq
  for(;;){}

  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1:2]), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[:]), seq
  for(;;){}

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[::]), seq
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ']'}}
  // expected-note@+2{{to match this '['}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[: :]), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copy(HasMem.MemArr[3:]), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc serial pcopy(HasMem.MemArr[3:])
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc serial present_or_copy(HasMem.MemArr[3:])
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'use_device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial use_device(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'use_device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial use_device(s.array[s.value : 5]), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial no_create(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial no_create(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial present(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial present(s.array[s.value : 5], s.value), seq
  for(;;){}


  void *IsPointer;
  // expected-error@+5{{expected ','}}
  // expected-error@+4{{expected pointer in 'deviceptr' clause, type is 'char'}}
  // expected-error@+3{{OpenACC sub-array is not allowed here}}
  // expected-note@+2{{expected variable of pointer type}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial deviceptr(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial deviceptr(IsPointer), seq
  for(;;){}

  // expected-error@+5{{expected ','}}
  // expected-error@+4{{expected pointer in 'attach' clause, type is 'char'}}
  // expected-error@+3{{OpenACC sub-array is not allowed here}}
  // expected-note@+2{{expected variable of pointer type}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial attach(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial attach(IsPointer), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'detach' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial detach(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'detach' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial detach(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial private(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial private(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial firstprivate(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial firstprivate(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'delete' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial delete(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'delete' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial delete(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'use_device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial use_device(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'use_device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial use_device(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'device_resident' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial device_resident(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'device_resident' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial device_resident(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'link' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial link(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'link' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial link(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'host' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial host(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'host' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial host(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{expected ','}}
  // expected-warning@+2{{OpenACC clause 'device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial device(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'device' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial device(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(zero:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc serial pcopyout(s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc serial present_or_copyout(zero:s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(zero : s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'zero'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(zero s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'readonly' on 'copyout' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(readonly:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'copyout' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'copyout' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'invalid'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyout(invalid s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(zero:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc serial pcreate(s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc serial present_or_create(zero:s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(zero : s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'zero'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(zero s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'readonly' on 'create' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(readonly:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'create' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'create' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'invalid'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial create(invalid s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(s.array[s.value] s.array[s.value :5] ), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(readonly:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc serial pcopyin(s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc serial present_or_copyin(readonly:s.array[s.value : 5], s.value)
  for(;;){}

  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(readonly : s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'readonly'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(readonly s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'zero' on 'copyin' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(zero :s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'copyin' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+2{{invalid tag 'invalid' on 'copyin' clause}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(invalid:s.array[s.value : 5], s.value), seq
  for(;;){}

  // expected-error@+3{{use of undeclared identifier 'invalid'}}
  // expected-error@+2{{expected ','}}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial copyin(invalid s.array[s.value : 5], s.value), seq
  for(;;){}
}

void ReductionClauseParsing() {
  char *Begin, *End;
  // expected-error@+1{{expected '('}}
#pragma acc serial reduction
  for(;;){}
  // expected-error@+2{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
  // expected-error@+1{{expected expression}}
#pragma acc serial reduction()
  for(;;){}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin)
  for(;;){}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin, End)
  for(;;){}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin, End)
  for(;;){}
#pragma acc serial reduction(+:Begin)
  for(;;){}
#pragma acc serial reduction(+:Begin, End)
  for(;;){}
#pragma acc serial reduction(*: Begin, End)
  for(;;){}
#pragma acc serial reduction(max : Begin, End)
  for(;;){}
#pragma acc serial reduction(min: Begin, End)
  for(;;){}
#pragma acc serial reduction(&: Begin, End)
  for(;;){}
#pragma acc serial reduction(|: Begin, End)
  for(;;){}
#pragma acc serial reduction(^: Begin, End)
  for(;;){}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial seq, reduction(&&: Begin, End)
  for(;;){}
  // expected-warning@+1{{OpenACC clause 'seq' not yet implemented, clause ignored}}
#pragma acc serial reduction(||: Begin, End), seq
  for(;;){}
}

int returns_int();

void IntExprParsing() {
  // expected-error@+1{{expected '('}}
#pragma acc parallel vector_length
  {}

  // expected-error@+1{{expected expression}}
#pragma acc parallel vector_length()
  {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel vector_length(invalid)
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel vector_length(5, 4)
  {}

#pragma acc parallel vector_length(5)
  {}

#pragma acc parallel vector_length(returns_int())
  {}

  // expected-error@+1{{expected '('}}
#pragma acc parallel num_gangs
  {}

  // expected-error@+1{{expected expression}}
#pragma acc parallel num_gangs()
  {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel num_gangs(invalid)
  {}

#pragma acc parallel num_gangs(5, 4)
  {}

#pragma acc parallel num_gangs(5)
  {}

#pragma acc parallel num_gangs(returns_int())
  {}

  // expected-error@+1{{expected '('}}
#pragma acc parallel num_workers
  {}

  // expected-error@+1{{expected expression}}
#pragma acc parallel num_workers()
  {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel num_workers(invalid)
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel num_workers(5, 4)
  {}

#pragma acc parallel num_workers(5)
  {}

#pragma acc parallel num_workers(returns_int())
  {}

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num()

  // expected-error@+2{{use of undeclared identifier 'invalid'}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num(invalid)

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num(5, 4)

  // expected-warning@+2{{OpenACC clause 'device_num' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num(5)

  // expected-warning@+2{{OpenACC clause 'device_num' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'init' not yet implemented, pragma ignored}}
#pragma acc init device_num(returns_int())

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async()

  // expected-error@+2{{use of undeclared identifier 'invalid'}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async(invalid)

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async(5, 4)

  // expected-warning@+2{{OpenACC clause 'default_async' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async(5)

  // expected-warning@+2{{OpenACC clause 'default_async' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'set' not yet implemented, pragma ignored}}
#pragma acc set default_async(returns_int())


  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector()
  // expected-error@+3{{invalid tag 'invalid' on 'vector' clause}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(invalid:)
  // expected-error@+3{{invalid tag 'invalid' on 'vector' clause}}
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(invalid:5)
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(length:)
  // expected-error@+3{{invalid tag 'num' on 'vector' clause}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(num:)
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(5, 4)
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(length:6,4)
  // expected-error@+4{{invalid tag 'num' on 'vector' clause}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(num:6,4)
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(5)
  // expected-error@+3{{invalid tag 'num' on 'vector' clause}}
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(num:5)
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(length:5)
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(returns_int())
  // expected-warning@+2{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop vector(length:returns_int())

  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker()
  // expected-error@+3{{invalid tag 'invalid' on 'worker' clause}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(invalid:)
  // expected-error@+3{{invalid tag 'invalid' on 'worker' clause}}
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(invalid:5)
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(num:)
  // expected-error@+3{{invalid tag 'length' on 'worker' clause}}
  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(length:)
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(5, 4)
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(num:6,4)
  // expected-error@+4{{invalid tag 'length' on 'worker' clause}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(length:6,4)
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(5)
  // expected-error@+3{{invalid tag 'length' on 'worker' clause}}
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(length:5)
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(num:5)
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(returns_int())
  // expected-error@+3{{invalid tag 'length' on 'worker' clause}}
  // expected-warning@+2{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop worker(length:returns_int())
}

void device_type() {
  // expected-error@+1{{expected '('}}
#pragma acc parallel device_type
  {}
  // expected-error@+1{{expected '('}}
#pragma acc parallel dtype
  {}

  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(
    {}
  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(
  {}

  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type()
  {}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel dtype()
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(*
  {}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(*
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(ident
  {}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(ident
  {}

  // expected-error@+3{{expected ','}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(ident ident2
  {}
  // expected-error@+3{{expected ','}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(ident ident2
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(ident, ident2
  {}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(ident, ident2
  {}

  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type(ident, ident2,)
  {}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel dtype(ident, ident2,)
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(*,)
  {}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(*,)
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel device_type(*,ident)
  {}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel dtype(*,ident)
  {}

  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type(ident, *)
  {}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel dtype(ident, *)
  {}

  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type("foo", 54)
  {}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel dtype(31, "bar")
  {}

#pragma acc parallel device_type(ident, auto, int, float)
  {}
#pragma acc parallel dtype(ident, auto, int, float)
  {}

#pragma acc parallel device_type(ident, auto, int, float) dtype(ident, auto, int, float)
  {}
}

#define acc_async_sync -1
void AsyncArgument() {
#pragma acc parallel async
  {}

  // expected-error@+1{{expected expression}}
#pragma acc parallel async()
  {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc parallel async(invalid)
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc parallel async(4, 3)
  {}

#pragma acc parallel async(returns_int())
  {}

#pragma acc parallel async(5)
  {}

#pragma acc parallel async(acc_async_sync)
  {}
}

void Tile() {

  int* Foo;
  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile
  for(;;){}
  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(
  for(;;){}
  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile()
  for(;;){}
  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(,
  for(;;){}
  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(,)
  for(;;){}
  // expected-error@+3{{use of undeclared identifier 'invalid'}}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(returns_int(), *, invalid, *)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(returns_int() *, Foo, *)
  for(;;){}

  // expected-error@+3{{indirection requires pointer operand ('int' invalid)}}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(* returns_int() , *)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(*)
  for(;;){}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(*Foo, *Foo)
  for(;;){}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(5)
  for(;;){}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(*, 5)
  for(;;){}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(5, *)
  for(;;){}
  // expected-warning@+2{{OpenACC clause 'tile' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop tile(5, *, 3, *)
  for(;;){}
}

void Gang() {
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang
  for(;;){}
  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(
  for(;;){}
  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang()
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(5, *)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(*)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(5, num:*)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num:5, *)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num:5, num:*)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num:*)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(dim:5)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(dim:5, dim:*)
  for(;;){}

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(dim:*)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*, static:5)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*, 5)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:45, 5)
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:45,
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:45
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*,
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(45,
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(45
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num:45,
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num:45
  for(;;){}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(dim:45,
  for(;;){}

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(dim:45
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(static:*, dim:returns_int(), 5)
  for(;;){}

  // expected-warning@+2{{OpenACC clause 'gang' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented, pragma ignored}}
#pragma acc loop gang(num: 32, static:*, dim:returns_int(), 5)
  for(;;){}

}

  // expected-warning@+5{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+4{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'nohost' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine worker, vector, seq, nohost
void bar();

  // expected-warning@+5{{OpenACC clause 'worker' not yet implemented, clause ignored}}
  // expected-warning@+4{{OpenACC clause 'vector' not yet implemented, clause ignored}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC clause 'nohost' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine(bar) worker, vector, seq, nohost


// Bind Clause Parsing.

  // expected-error@+2{{expected '('}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine bind
void BCP1();

  // expected-error@+2{{expected identifier or string literal}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine(BCP1) bind()

  // expected-warning@+2{{OpenACC clause 'bind' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine bind("ReductionClauseParsing")
void BCP2();

  // expected-warning@+2{{OpenACC clause 'bind' not yet implemented, clause ignored}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine(BCP1) bind(BCP2)

  // expected-error@+2{{use of undeclared identifier 'unknown_thing'}}
  // expected-warning@+1{{OpenACC construct 'routine' not yet implemented, pragma ignored}}
#pragma acc routine(BCP1) bind(unknown_thing)
