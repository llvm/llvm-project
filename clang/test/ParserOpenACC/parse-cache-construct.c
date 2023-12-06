// RUN: %clang_cc1 %s -verify -fopenacc

char *getArrayPtr();
void func() {
  char Array[10];
  char *ArrayPtr = getArrayPtr();
  int *readonly;

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected '('}}
    // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache clause list
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache()
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache() clause-list
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{use of undeclared identifier 'invalid'}}
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(invalid
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{expected '['}}
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{use of undeclared identifier 'invalid'}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(invalid)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '['}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{expected expression}}
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr[
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(ArrayPtr[, 5)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(Array[)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(Array[*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{expected expression}}
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(Array[*readonly:
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '['}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '['}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected expression}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected '['}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly], Array)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly], Array[*readonly:3])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5 + i:*readonly], Array[*readonly + i:3])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+4{{expected identifier}}
    // expected-error@+3{{expected ')'}}
    // expected-note@+2{{to match this '('}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly],
  }

  for (int i = 0; i < 10; ++i) {
    // expected-error@+2{{expected identifier}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:*readonly],)
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+2{{left operand of comma operator has no effect}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5,6:*readonly])
  }

  for (int i = 0; i < 10; ++i) {
    // expected-warning@+2{{left operand of comma operator has no effect}}
    // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
    #pragma acc cache(readonly:ArrayPtr[5:3, *readonly], ArrayPtr[0])
  }

}
