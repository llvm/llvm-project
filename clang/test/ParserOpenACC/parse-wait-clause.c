// RUN: %clang_cc1 %s -verify -fopenacc

void func() {
  int i, j;

  #pragma acc parallel wait
  {}

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait clause-list
  {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (
      {}

  #pragma acc parallel wait ()
      {}

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait () clause-list
      {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (devnum:
    {}

  // expected-error@+1{{expected expression}}
  #pragma acc parallel wait (devnum:)
    {}

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait (devnum:) clause-list
    {}

  // expected-error@+3{{expected ':'}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (devnum: i + j
    {}

  // expected-error@+1{{expected ':'}}
  #pragma acc parallel wait (devnum: i + j)
    {}

  // expected-error@+2{{expected ':'}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait (devnum: i + j) clause-list
    {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (queues:
    {}

  #pragma acc parallel wait (queues:)
    {}

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait (queues:) clause-list
    {}

  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (devnum: i + j:queues:
    {}

  #pragma acc parallel wait (devnum: i + j:queues:)
    {}

  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait (devnum: i + j:queues:) clause-list
    {}

  // expected-error@+4{{use of undeclared identifier 'devnum'}}
  // expected-error@+3{{expected ','}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait (queues:devnum: i + j
    {}

  // expected-error@+2{{expected ','}}
  // expected-error@+1{{use of undeclared identifier 'devnum'}}
  #pragma acc parallel wait (queues:devnum: i + j)
    {}

  // expected-error@+3{{expected ','}}
  // expected-error@+2{{use of undeclared identifier 'devnum'}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait (queues:devnum: i + j) clause-list
    {}

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(i, j, 1+1, 3.3
    {}

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc parallel wait(i, j, 1+1, 3.3)
    {}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait(i, j, 1+1, 3.3) clause-list
    {}

  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(,
    {}

  // expected-error@+1{{expected expression}}
  #pragma acc parallel wait(,)
    {}

  // expected-error@+2{{expected expression}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait(,) clause-list
    {}

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(queues:i, j, 1+1, 3.3
    {}

  // expected-error@+4{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+3{{expected expression}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(queues:i, j, 1+1, 3.3,
    {}

  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc parallel wait(queues:i, j, 1+1, 3.3)
    {}

  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait(queues:i, j, 1+1, 3.3) clause-list
    {}

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(devnum:3:i, j, 1+1, 3.3
    {}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc parallel wait(devnum:3:i, j, 1+1, 3.3)
    {}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait(devnum:3:i, j, 1+1, 3.3) clause-list
    {}

  // expected-error@+3{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
  #pragma acc parallel wait(devnum:3:queues:i, j, 1+1, 3.3
    {}
  // expected-error@+1{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  #pragma acc parallel wait(devnum:3:queues:i, j, 1+1, 3.3)
    {}
  // expected-error@+2{{OpenACC clause 'wait' requires expression of integer type ('double' invalid)}}
  // expected-error@+1{{invalid OpenACC clause 'clause'}}
  #pragma acc parallel wait(devnum:3:queues:i, j, 1+1, 3.3) clause-list
    {}
}
