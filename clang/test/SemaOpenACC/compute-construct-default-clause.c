// RUN: %clang_cc1 %s -fopenacc -verify

void SingleOnly() {
  #pragma acc parallel default(none)
  while(0);

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'serial' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc serial default(present) self default(none)
  while(0);

  int i;

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc kernels self default(present) present(i) default(none) copy(i)
  while(0);

  // expected-warning@+6{{OpenACC construct 'parallel loop' not yet implemented}}
  // expected-warning@+5{{OpenACC clause 'self' not yet implemented}}
  // expected-warning@+4{{OpenACC clause 'default' not yet implemented}}
  // expected-warning@+3{{OpenACC clause 'private' not yet implemented}}
  // expected-warning@+2{{OpenACC clause 'default' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'copy' not yet implemented}}
  #pragma acc parallel loop self default(present) private(i) default(none) copy(i)
  while(0);

  // expected-warning@+3{{OpenACC clause 'self' not yet implemented, clause ignored}}
  // expected-warning@+2{{OpenACC construct 'serial loop' not yet implemented}}
  // expected-error@+1{{expected '('}}
  #pragma acc serial loop self default private(i) default(none) if(i)
  while(0);

  // expected-warning@+2{{OpenACC construct 'kernels loop' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented}}
  #pragma acc kernels loop default(none)
  while(0);

  // expected-warning@+2{{OpenACC construct 'data' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented}}
  #pragma acc data default(none)
  while(0);

  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
  #pragma acc loop default(none)
  for(;;);

  // expected-warning@+2{{OpenACC construct 'wait' not yet implemented}}
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'wait' directive}}
  #pragma acc wait default(none)
  while(0);

  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(present)
  for(;;);
}
