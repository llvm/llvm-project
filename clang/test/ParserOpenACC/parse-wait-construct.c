// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
  int i, j;

  #pragma acc wait

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait clause-list

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (

  #pragma acc wait ()

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait () clause-list

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (devnum:

  // expected-error@+1{{expected expression}}
  #pragma acc wait (devnum:)

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait (devnum:) clause-list

  // expected-error@+3{{expected ':'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (devnum: i + j

  // expected-error@+1{{expected ':'}}
  #pragma acc wait (devnum: i + j)

  // expected-error@+2{{expected ':'}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait (devnum: i + j) clause-list

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (queues:

  #pragma acc wait (queues:)

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait (queues:) clause-list

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (devnum: i + j:queues:

  #pragma acc wait (devnum: i + j:queues:)

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait (devnum: i + j:queues:) clause-list

  // expected-error@+4{{use of undeclared identifier 'devnum'}}
  // expected-error@+3{{expected ','}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait (queues:devnum: i + j

  // expected-error@+2{{use of undeclared identifier 'devnum'}}
  // expected-error@+1{{expected ','}}
  #pragma acc wait (queues:devnum: i + j)

  // expected-error@+3{{use of undeclared identifier 'devnum'}}
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait (queues:devnum: i + j) clause-list

  // expected-error@+3{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(i, j, 1+1, 3.3

  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc wait(i, j, 1+1, 3.3)
  // expected-error@+2{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait(i, j, 1+1, 3.3) clause-list

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(,

  // expected-error@+1{{expected expression}}
  #pragma acc wait(,)

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait(,) clause-list

  // expected-error@+3{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(queues:i, j, 1+1, 3.3

  // expected-error@+4{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(queues:i, j, 1+1, 3.3,

  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc wait(queues:i, j, 1+1, 3.3)

  // expected-error@+2{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait(queues:i, j, 1+1, 3.3) clause-list

  // expected-error@+3{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3
  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3)
  // expected-error@+2{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait(devnum:3:i, j, 1+1, 3.3) clause-list

  // expected-error@+3{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3
  // expected-error@+1{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3)
  // expected-error@+2{{OpenACC directive 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc wait(devnum:3:queues:i, j, 1+1, 3.3) clause-list
}
