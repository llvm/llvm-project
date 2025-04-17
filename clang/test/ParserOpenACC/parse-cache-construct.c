// RUN: %clang_cc1 %s -verify -fopenacc

struct S {
  int foo;
  char Array[1];
};
char *getArrayPtr();
void func() {
  char Array[10];
  char *ArrayPtr = getArrayPtr();
  int *readonly;
  struct S s;

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{expected '('}}
    #pragma acc cache
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '('}}
    // expected-error@+1{{invalid OpenACC clause 'clause'}}
    #pragma acc cache clause list
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{expected expression}}
    #pragma acc cache()
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-error@+1{{invalid OpenACC clause 'clause'}}
    #pragma acc cache() clause-list
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected ')'}}
    // expected-note@+1{{to match this '('}}
    #pragma acc cache(
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{use of undeclared identifier 'invalid'}}
    // expected-error@+2{{expected ')'}}
    // expected-note@+1{{to match this '('}}
    #pragma acc cache(invalid
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(ArrayPtr
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{use of undeclared identifier 'invalid'}}
    #pragma acc cache(invalid)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+5{{expected expression}}
    // expected-error@+4{{expected ']'}}
    // expected-note@+3{{to match this '['}}
    // expected-error@+2{{expected ')'}}
    // expected-note@+1{{to match this '('}}
    #pragma acc cache(ArrayPtr[
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected expression}}
    // expected-error@+2{{expected ']'}}
    // expected-note@+1{{to match this '['}}
    #pragma acc cache(ArrayPtr[, 5)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected expression}}
    // expected-error@+2{{expected ']'}}
    // expected-note@+1{{to match this '['}}
    #pragma acc cache(Array[)
  }

  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(Array[*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+5{{expected expression}}
    // expected-error@+4{{expected ']'}}
    // expected-note@+3{{to match this '['}}
    // expected-error@+2{{expected ')'}}
    // expected-note@+1{{to match this '('}}
    #pragma acc cache(Array[*readonly:
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(readonly)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{invalid tag 'devnum' on 'cache' directive}}
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(devnum:ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{invalid tag 'invalid' on 'cache' directive}}
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(invalid:ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(readonly:ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(readonly:ArrayPtr[5:1])
  }

  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(readonly:ArrayPtr[5:*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly], Array)
  }

  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(readonly:ArrayPtr[5:*readonly], Array[*readonly:3])
  }

  for (int i = 0; i < 10; ++i) {
    #pragma acc cache(readonly:ArrayPtr[5 + i:*readonly], Array[*readonly + i:3])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected expression}}
    // expected-error@+2{{expected ')'}}
    // expected-note@+1{{to match this '('}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly],
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+1{{expected expression}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly],)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{left operand of comma operator has no effect}}
    #pragma acc cache(readonly:ArrayPtr[5,6:*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{left operand of comma operator has no effect}}
    #pragma acc cache(readonly:ArrayPtr[5:3, *readonly], ArrayPtr[0])
  }

  for (int i = 0; i < 10; ++i) {
  // expected-error@+1{{OpenACC variable in cache directive is not a valid sub-array or array element}}
    #pragma acc cache(readonly:s.foo)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{left operand of comma operator has no effect}}
    #pragma acc cache(readonly:s.Array[1,2])
  }
}
