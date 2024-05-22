// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
  int i, j;

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait ()

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait () clause-list

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum:

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum:)

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum:) clause-list

  // expected-error@+4{{expected ':'}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j

  // expected-error@+2{{expected ':'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j)

  // expected-error@+3{{expected ':'}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:)

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j:queues:

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j:queues:)

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (devnum: i + j:queues:) clause-list

  // expected-error@+4{{use of undeclared identifier 'devnum'}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:devnum: i + j

  // expected-error@+2{{use of undeclared identifier 'devnum'}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:devnum: i + j)

  // expected-error@+3{{use of undeclared identifier 'devnum'}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait (queues:devnum: i + j) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(i, j, 1+1, 3.3

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(i, j, 1+1, 3.3)
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(i, j, 1+1, 3.3) clause-list

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(,

  // expected-error@+2{{expected expression}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(,)

  // expected-error@+3{{expected expression}}
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(,) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(queues:i, j, 1+1, 3.3

  // expected-error@+4{{expected expression}}
  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(queues:i, j, 1+1, 3.3,

  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(queues:i, j, 1+1, 3.3)

  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(queues:i, j, 1+1, 3.3) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3)
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3) clause-list

  // expected-error@+3{{expected ')'}}
  // expected-note@+2{{to match this '('}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3)
  // expected-warning@+2{{OpenACC clause parsing not yet implemented}}
  // expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3) clause-list
}
