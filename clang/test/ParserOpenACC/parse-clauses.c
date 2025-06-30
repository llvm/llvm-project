// RUN: %clang_cc1 %s -verify=expected,c -fopenacc -std=c99
// RUN: %clang_cc1 %s -verify=expected,c -fopenacc
// RUN: %clang_cc1 %s -verify=expected,cpp -fopenacc -x c++

void func() {

  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete', or 'detach' clause}}
#pragma acc exit data finalize

  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete', or 'detach' clause}}
#pragma acc exit data finalize finalize

  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete', or 'detach' clause}}
  // expected-error@+1{{invalid OpenACC clause 'invalid'}}
#pragma acc exit data finalize invalid

  // expected-error@+2{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete', or 'detach' clause}}
  // expected-error@+1{{invalid OpenACC clause 'invalid'}}
#pragma acc exit data finalize invalid invalid finalize

  // expected-error@+1{{OpenACC 'exit data' construct must have at least one 'copyout', 'delete', or 'detach' clause}}
#pragma acc exit data wait finalize

  // expected-error@+1{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
#pragma acc host_data if_present

  // expected-error@+1{{OpenACC 'host_data' construct must have at least one 'use_device' clause}}
#pragma acc host_data if_present, if_present

  // expected-error@+4{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+3{{previous 'seq' clause is here}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc loop seq independent auto
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+3{{previous 'seq' clause is here}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc loop seq, independent auto
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+3{{previous 'seq' clause is here}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'loop' construct}}
  // expected-note@+1{{previous 'seq' clause is here}}
#pragma acc loop seq independent, auto
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'kernels loop' construct}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'kernels loop' construct}}
  // expected-note@+1 2{{previous 'seq' clause is here}}
#pragma acc kernels loop seq independent auto
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'serial loop' construct}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'serial loop' construct}}
  // expected-note@+1 2{{previous 'seq' clause is here}}
#pragma acc serial loop seq, independent auto
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{OpenACC clause 'independent' may not appear on the same construct as a 'seq' clause on a 'parallel loop' construct}}
  // expected-error@+2{{OpenACC clause 'auto' may not appear on the same construct as a 'seq' clause on a 'parallel loop' construct}}
  // expected-note@+1 2{{previous 'seq' clause is here}}
#pragma acc parallel loop seq independent, auto
  for(int i = 0; i < 5;++i) {}


  // expected-error@+1{{expected identifier}}
#pragma acc loop , seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc loop seq,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc loop collapse
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop collapse()
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{invalid tag 'unknown' on 'collapse' clause}}
  // expected-error@+1{{expected expression}}
#pragma acc loop collapse(unknown:)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop collapse(force:)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{invalid tag 'unknown' on 'collapse' clause}}
#pragma acc loop collapse(unknown:1)
  for(int i = 0; i < 5;++i) {}

#pragma acc loop collapse(force:1)
  for(int i = 0; i < 5;++i) {}

#pragma acc loop collapse(1)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop collapse(5, 6)
  for(int i = 0; i < 5;++i) {}
}

void DefaultClause() {
  // expected-error@+1{{expected '('}}
#pragma acc serial loop default
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc serial default self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc serial default, self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default(
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{invalid value for 'default' clause; expected 'present' or 'none'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default( self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected identifier}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial default(, self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial default)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial default), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial default()
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial default() self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial default(), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{invalid value for 'default' clause; expected 'present' or 'none'}}
#pragma acc serial default(invalid)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{invalid value for 'default' clause; expected 'present' or 'none'}}
#pragma acc serial default(auto) self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{invalid value for 'default' clause; expected 'present' or 'none'}}
#pragma acc serial default(invalid), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial default(none)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial default(present), self
  for(int i = 0; i < 5;++i) {}
}

void IfClause() {
  int i, j;
  // expected-error@+1{{expected '('}}
#pragma acc serial loop if
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc serial if private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc serial if, private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if(
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{use of undeclared identifier 'self'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if( self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{use of undeclared identifier 'self'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial if(, self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if) private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial if), private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial if()
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial if() private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial if(), private(i)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'invalid_expr'}}
#pragma acc serial if(invalid_expr)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial if() private(i)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial if(i > j)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial if(1+5>3), private(i)
  for(int i = 0; i < 5;++i) {}
}

void SelfClause() {
#pragma acc serial loop self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial loop self, seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial loop self(
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial loop self( seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial loop self(, seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial loop self)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial loop self) seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected identifier}}
#pragma acc serial loop self), seq
  for(int i = 0; i < 5;++i) {}


  // expected-error@+1{{expected expression}}
#pragma acc serial loop self(), seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{expected expression}}
#pragma acc serial loop self(,), seq
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'invalid_expr'}}
#pragma acc serial loop self(invalid_expr), seq
  for(int i = 0; i < 5;++i) {}

  int i, j;

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial self(i > j
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{use of undeclared identifier 'seq'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial self(i > j, seq
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{left operand of comma operator has no effect}}
#pragma acc serial self(i, j)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial self(i > j)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial self(1+5>3), private(i)
  for(int i = 0; i < 5;++i) {}
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

  // expected-error@+1{{expected '('}}
#pragma acc update host(s) self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'zero'}}
#pragma acc update self(zero : s.array[s.value : 5], s.value), if_present
  for(int i = 0; i < 5;++i) {}

#pragma acc update self(s.array[s.value : 5], s.value), if_present
  for(int i = 0; i < 5;++i) {}
}

void VarListClauses() {
  // expected-error@+1{{expected '('}}
#pragma acc serial copy
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected '('}}
#pragma acc serial copy, self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial copy)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{expected identifier}}
#pragma acc serial copy), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial copy(
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc serial copy(, self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial copy()
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc serial copy(), self
  for(int i = 0; i < 5;++i) {}

  struct Members s;
  struct HasMembersArray HasMem;

#pragma acc serial copy(s.array[s.value]), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copy(s.array[s.value], s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copy(HasMem.MemArr[3].array[1]), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copy(HasMem.MemArr[3].array[1:4]), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1]), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC sub-array is not allowed here}}
#pragma acc serial copy(HasMem.MemArr[1:3].array[1:2]), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copy(HasMem.MemArr[:]), self
  for(int i = 0; i < 5;++i) {}

  // cpp-error@+2{{expected unqualified-id}}
  // c-error@+1{{expected expression}}
#pragma acc serial copy(HasMem.MemArr[::]), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ']'}}
  // expected-note@+1{{to match this '['}}
#pragma acc serial copy(HasMem.MemArr[: :]), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copy(HasMem.MemArr[3:]), self
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'pcopy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc serial pcopy(HasMem.MemArr[3:])
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'present_or_copy' is a deprecated clause name and is now an alias for 'copy'}}
#pragma acc serial present_or_copy(HasMem.MemArr[3:])
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{unknown modifier 'foo' in OpenACC modifier-list on 'copy' clause}}
  // expected-error@+1{{unknown modifier 'bar' in OpenACC modifier-list on 'copy' clause}}
#pragma acc parallel copy(foo, bar: HasMem.MemArr[3:]) self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{duplicate modifier 'always' in OpenACC modifier-list on 'copy' clause}}
#pragma acc parallel copy(always, alwaysin, always: HasMem.MemArr[3:]) self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'always'}}
#pragma acc parallel copy(always, alwaysin, always, HasMem.MemArr[3:]) self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2 2{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
  // expected-error@+1{{expected ','}}
#pragma acc host_data use_device(s.array[s.value] s.array[s.value :5] ), if_present
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC variable in 'use_device' clause is not a valid variable name or array name}}
#pragma acc host_data use_device(s.array[s.value : 5]), if_present
  for(int i = 0; i < 5;++i) {}

#pragma acc host_data use_device(HasMem), if_present
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial no_create(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial no_create(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial present(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial present(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}


  void *IsPointer;
  // expected-error@+4{{expected ','}}
  // expected-error@+3{{expected pointer in 'deviceptr' clause, type is 'char'}}
  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc serial deviceptr(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial deviceptr(IsPointer), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{expected ','}}
  // expected-error@+3{{expected pointer in 'attach' clause, type is 'char'}}
  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc serial attach(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial attach(IsPointer), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+4{{expected ','}}
  // expected-error@+3{{expected pointer in 'detach' clause, type is 'char'}}
  // expected-error@+2{{OpenACC sub-array is not allowed here}}
  // expected-note@+1{{expected variable of pointer type}}
#pragma acc exit data copyout(s) detach(s.array[s.value] s.array[s.value :5])

#pragma acc exit data copyout(s) detach(IsPointer)

  // expected-error@+1{{expected ','}}
#pragma acc serial private(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial private(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial firstprivate(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial firstprivate(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc exit data delete(s.array[s.value] s.array[s.value :5] ) async
  for(int i = 0; i < 5;++i) {}

#pragma acc exit data delete(s.array[s.value : 5], s.value),async
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected ','}}
  // expected-error@+2{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
  // expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare device_resident(s.array[s.value] s.array[s.value :5] ), copy(s)

  int CopyRef1, CopyRef2, CopyRef3;

  // expected-error@+2{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
  // expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare device_resident(s.array[s.value : 5], s.value), copy(CopyRef1)

  // expected-error@+3{{expected ','}}
  // expected-error@+2{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
  // expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare link(s.array[s.value] s.array[s.value :5] ), copy(CopyRef2)

  // expected-error@+2{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
  // expected-error@+1{{OpenACC variable on 'declare' construct is not a valid variable name or array name}}
#pragma acc declare link(s.array[s.value : 5], s.value), copy(CopyRef3)

  // expected-error@+1{{expected ','}}
#pragma acc update host(s.array[s.value] s.array[s.value :5] )
  for(int i = 0; i < 5;++i) {}

#pragma acc update host(s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc update device(s.array[s.value] s.array[s.value :5] )
  for(int i = 0; i < 5;++i) {}

#pragma acc update device(s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial copyout(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyout(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyout(zero:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'pcopyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc serial pcopyout(s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'present_or_copyout' is a deprecated clause name and is now an alias for 'copyout'}}
#pragma acc serial present_or_copyout(zero:s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyout(zero : s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'zero'}}
#pragma acc serial copyout(zero s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'copyout' clause}}
#pragma acc serial copyout(readonly:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'copyout' clause}}
#pragma acc serial copyout(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'copyout' clause}}
#pragma acc serial copyout(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc serial copyout(invalid s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{unknown modifier 'invalid' in OpenACC modifier-list on 'copyout' clause}}
  // expected-error@+1{{unknown modifier 'bar' in OpenACC modifier-list on 'copyout' clause}}
#pragma acc serial copyout(invalid, bar: s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{duplicate modifier 'zero' in OpenACC modifier-list on 'copyout' clause}}
#pragma acc serial copyout(zero, zero, always: s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial create(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial create(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial create(zero:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC 'always' modifier not valid on 'create' clause}}
#pragma acc serial create(always, zero:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{duplicate modifier 'always' in OpenACC modifier-list on 'create' clause}}
  // expected-error@+1{{OpenACC 'always' modifier not valid on 'create' clause}}
#pragma acc serial create(always, always, zero:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{unknown modifier 'invalid' in OpenACC modifier-list on 'create' clause}}
  // expected-error@+1{{OpenACC 'always' modifier not valid on 'create' clause}}
#pragma acc serial create(always, invalid, zero:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'pcreate' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc serial pcreate(s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'present_or_create' is a deprecated clause name and is now an alias for 'create'}}
#pragma acc serial present_or_create(zero:s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial create(zero : s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'zero'}}
#pragma acc serial create(zero s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC 'readonly' modifier not valid on 'create' clause}}
#pragma acc serial create(readonly:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'create' clause}}
#pragma acc serial create(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'create' clause}}
#pragma acc serial create(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc serial create(invalid s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected ','}}
#pragma acc serial copyin(s.array[s.value] s.array[s.value :5] ), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyin(s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyin(readonly:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'pcopyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc serial pcopyin(s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

  // expected-warning@+1{{OpenACC clause name 'present_or_copyin' is a deprecated clause name and is now an alias for 'copyin'}}
#pragma acc serial present_or_copyin(readonly:s.array[s.value : 5], s.value)
  for(int i = 0; i < 5;++i) {}

#pragma acc serial copyin(readonly : s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'readonly'}}
#pragma acc serial copyin(readonly s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{OpenACC 'zero' modifier not valid on 'copyin' clause}}
#pragma acc serial copyin(zero :s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'copyin' clause}}
#pragma acc serial copyin(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{unknown modifier 'invalid' in OpenACC modifier-list on 'copyin' clause}}
#pragma acc serial copyin(invalid:s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc serial copyin(invalid s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{unknown modifier 'foo' in OpenACC modifier-list on 'copyin' clause}}
  // expected-error@+1{{unknown modifier 'bar' in OpenACC modifier-list on 'copyin' clause}}
#pragma acc serial copyin(foo, bar: s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{duplicate modifier 'readonly' in OpenACC modifier-list on 'copyin' clause}}
#pragma acc serial copyin(always, readonly, readonly: s.array[s.value : 5], s.value), self
  for(int i = 0; i < 5;++i) {}
}

void ReductionClauseParsing() {
  char *Begin, *End;
  // expected-error@+1{{expected '('}}
#pragma acc serial reduction
  for(int i = 0; i < 5;++i) {}
  // expected-error@+2{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
  // expected-error@+1{{expected expression}}
#pragma acc serial reduction()
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin)
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin, End)
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{missing reduction operator, expected '+', '*', 'max', 'min', '&', '|', '^', '&&', or '||', follwed by a ':'}}
#pragma acc serial reduction(Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(+:Begin)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(+:Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(*: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(max : Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(min: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(&: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(|: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(^: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial self, reduction(&&: Begin, End)
  for(int i = 0; i < 5;++i) {}
#pragma acc serial reduction(||: Begin, End), self
  for(int i = 0; i < 5;++i) {}
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

  // expected-error@+1{{expected '('}}
#pragma acc init device_num

  // expected-error@+1{{expected expression}}
#pragma acc init device_num()

  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc init device_num(invalid)

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc init device_num(5, 4)

#pragma acc init device_num(5)

#pragma acc init device_num(returns_int())

  // expected-error@+2{{expected '('}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', or 'device_type' clause}}
#pragma acc set default_async

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', or 'device_type' clause}}
#pragma acc set default_async()

  // expected-error@+2{{use of undeclared identifier 'invalid'}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', or 'device_type' clause}}
#pragma acc set default_async(invalid)

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-error@+1{{OpenACC 'set' construct must have at least one 'default_async', 'device_num', or 'device_type' clause}}
#pragma acc set default_async(5, 4)

#pragma acc set default_async(5)

#pragma acc set default_async(returns_int())


#pragma acc loop vector
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop vector()
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{invalid tag 'invalid' on 'vector' clause}}
  // expected-error@+1{{expected expression}}
#pragma acc loop vector(invalid:)
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{invalid tag 'invalid' on 'vector' clause}}
#pragma acc loop vector(invalid:5)
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop vector(length:)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{invalid tag 'num' on 'vector' clause}}
  // expected-error@+1{{expected expression}}
#pragma acc loop vector(num:)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop vector(5, 4)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop vector(length:6,4)
  for(int i = 0; i < 5;++i);
  // expected-error@+3{{invalid tag 'num' on 'vector' clause}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop vector(num:6,4)
  for(int i = 0; i < 5;++i);
#pragma acc loop vector(5)
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{invalid tag 'num' on 'vector' clause}}
#pragma acc loop vector(num:5)
  for(int i = 0; i < 5;++i);
#pragma acc loop vector(length:5)
  for(int i = 0; i < 5;++i);
#pragma acc loop vector(returns_int())
  for(int i = 0; i < 5;++i);
#pragma acc loop vector(length:returns_int())
  for(int i = 0; i < 5;++i);

#pragma acc loop worker
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop worker()
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{invalid tag 'invalid' on 'worker' clause}}
  // expected-error@+1{{expected expression}}
#pragma acc loop worker(invalid:)
  for(int i = 0; i < 5;++i);
#pragma acc kernels
  // expected-error@+1{{invalid tag 'invalid' on 'worker' clause}}
#pragma acc loop worker(invalid:5)
  for(int i = 0; i < 5;++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop worker(num:)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{invalid tag 'length' on 'worker' clause}}
  // expected-error@+1{{expected expression}}
#pragma acc loop worker(length:)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop worker(5, 4)
  for(int i = 0; i < 5;++i);
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop worker(num:6,4)
  for(int i = 0; i < 5;++i);
  // expected-error@+3{{invalid tag 'length' on 'worker' clause}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop worker(length:6,4)
  for(int i = 0; i < 5;++i);
#pragma acc kernels
#pragma acc loop worker(5)
  for(int i = 0; i < 5;++i);
#pragma acc kernels
  // expected-error@+1{{invalid tag 'length' on 'worker' clause}}
#pragma acc loop worker(length:5)
  for(int i = 0; i < 5;++i);
#pragma acc kernels
#pragma acc loop worker(num:5)
  for(int i = 0; i < 5;++i);
#pragma acc kernels
#pragma acc loop worker(returns_int())
  for(int i = 0; i < 5;++i);
#pragma acc kernels
  // expected-error@+1{{invalid tag 'length' on 'worker' clause}}
#pragma acc loop worker(length:returns_int())
  for(int i = 0; i < 5;++i);
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

  // expected-error@+4{{invalid value 'ident' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+3{{invalid value 'auto' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+2{{invalid value 'int' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+1{{invalid value 'float' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc parallel device_type(ident, auto, int, float)
  {}
  // expected-error@+4{{invalid value 'ident' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+3{{invalid value 'auto' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+2{{invalid value 'int' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+1{{invalid value 'float' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc parallel dtype(ident, auto, int, float)
  {}

  // expected-error@+8{{invalid value 'ident' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+7{{invalid value 'auto' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+6{{invalid value 'int' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+5{{invalid value 'float' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+4{{invalid value 'ident' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+3{{invalid value 'auto' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+2{{invalid value 'int' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+1{{invalid value 'float' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
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
  // expected-error@+1{{expected '('}}
#pragma acc loop tile
  for(int i = 0; i < 5;++i) {}
  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop tile(
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{expected expression}}
#pragma acc loop tile()
  for(int i = 0; i < 5;++i) {}
  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop tile(,
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{expected expression}}
#pragma acc loop tile(,)
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{use of undeclared identifier 'invalid'}}
#pragma acc loop tile(returns_int(), *, invalid, *)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop tile(returns_int() *, Foo, *)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{indirection requires pointer operand ('int' invalid)}}
#pragma acc loop tile(* returns_int() , *)
  for(int j = 0; j < 5;++j){
    for(int i = 0; i < 5;++i);
  }

#pragma acc loop tile(*)
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{OpenACC 'tile' clause size expression must be an asterisk or a constant expression}}
#pragma acc loop tile(*Foo, *Foo)
  for(int i = 0; i < 5;++i) {}
#pragma acc loop tile(5)
  for(int i = 0; i < 5;++i) {}
#pragma acc loop tile(*, 5)
  for(int j = 0; j < 5;++j){
    for(int i = 0; i < 5;++i);
  }
#pragma acc loop tile(5, *)
  for(int j = 0; j < 5;++j){
    for(int i = 0; i < 5;++i);
  }
#pragma acc loop tile(5, *, 3, *)
  for(int j = 0; j < 5;++j){
    for(int k = 0; k < 5;++k)
      for(int l = 0;l < 5;++l)
        for(int i = 0; i < 5;++i);
  }
}

void Gang() {
#pragma acc loop gang
  for(int i = 0; i < 5;++i) {}
  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(
  for(int i = 0; i < 5;++i) {}
  // expected-error@+1{{expected expression}}
#pragma acc loop gang()
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(5, *)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(*)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(5, num:*)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(num:5, *)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(num:5, num:*)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(num:*)
  for(int i = 0; i < 5;++i) {}

#pragma acc loop gang(dim:2)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(dim:5, dim:*)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{expected expression}}
#pragma acc loop gang(dim:*)
  for(int i = 0; i < 5;++i) {}

#pragma acc loop gang(static:*)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{OpenACC 'gang' clause may have at most one 'static' argument}}
  // expected-note@+1{{previous expression is here}}
#pragma acc loop gang(static:*, static:5)
  for(int i = 0; i < 5;++i) {}

#pragma acc kernels
#pragma acc loop gang(static:*, 5)
  for(int i = 0; i < 5;++i) {}

#pragma acc kernels
#pragma acc loop gang(static:45, 5)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(static:45,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(static:45
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(static:*,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(static:*
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(45,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(45
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(num:45,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(num:45
  for(int i = 0; i < 5;++i) {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(dim:45,
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc loop gang(dim:45
  for(int i = 0; i < 5;++i) {}

#pragma acc kernels
#pragma acc loop gang(static:*, 5)
  for(int i = 0; i < 5;++i) {}

  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc loop gang(static:*, dim:returns_int())
  for(int i = 0; i < 5;++i) {}

  // expected-error@+2 2{{'num' argument on 'gang' clause is not permitted on an orphaned 'loop' construct}}
  // expected-error@+1{{argument to 'gang' clause dimension must be a constant expression}}
#pragma acc loop gang(num: 32, static:*, dim:returns_int(), 5)
  for(int i = 0; i < 5;++i) {}

}

  // expected-error@+4{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+3{{previous 'worker' clause is here}}
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine worker, vector, seq, nohost
void bar();

  // expected-error@+4{{OpenACC clause 'seq' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+3{{previous 'worker' clause is here}}
  // expected-error@+2{{OpenACC clause 'vector' may not appear on the same construct as a 'worker' clause on a 'routine' construct}}
  // expected-note@+1{{previous 'worker' clause is here}}
#pragma acc routine(bar) worker, vector, seq, nohost


// Bind Clause Parsing.

// expected-error@+1{{expected '('}}
#pragma acc routine seq bind
void BCP1();

  // expected-error@+1{{expected identifier or string literal in OpenACC 'bind' clause}}
#pragma acc routine(BCP1) seq bind()

  // expected-error@+1{{expected identifier or string literal in OpenACC 'bind' clause}}
#pragma acc routine(BCP1) seq bind(1)

  // expected-error@+1{{expected identifier or string literal in OpenACC 'bind' clause}}
#pragma acc routine(BCP1) gang bind(0xF)

// expected-error@+1{{expected function or lambda declaration for 'routine' construct}}
#pragma acc routine seq bind("ReductionClauseParsing")

#pragma acc routine(BCP1) seq bind(unknown_thing)

void AtomicIf() {
  int i, j;
  // expected-error@+1{{expected '('}}
#pragma acc atomic read if
  i = j;
#pragma acc atomic read if (0)
  i = j;
#pragma acc atomic write if (1)
  i = j + 1;

#pragma acc atomic update if (i)
  ++i;
#pragma acc atomic if (j)
  ++i;

#pragma acc atomic capture if (0)
  i = j++;
#pragma acc atomic capture if (i)
  {
    ++j;
    i = j;
  }
}
