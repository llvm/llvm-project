// RUN: %clang_cc1 %s -fopenacc -verify

void SingleOnly() {
  #pragma acc parallel default(none)
  while(0);

  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'serial' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc serial default(present) seq default(none)
  while(0);

  // expected-warning@+5{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc kernels seq default(present) seq default(none) seq
  while(0);

  // expected-warning@+6{{OpenACC construct 'parallel loop' not yet implemented}}
  // expected-warning@+5{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+2{{OpenACC clause 'default' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented}}
  #pragma acc parallel loop seq default(present) seq default(none) seq
  while(0);

  // expected-warning@+3{{OpenACC construct 'serial loop' not yet implemented}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+1{{expected '('}}
  #pragma acc serial loop seq default seq default(none) seq
  while(0);

  // expected-warning@+2{{OpenACC construct 'kernels loop' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented}}
  #pragma acc kernels loop default(none)
  while(0);

  // expected-warning@+2{{OpenACC construct 'data' not yet implemented}}
  // expected-warning@+1{{OpenACC clause 'default' not yet implemented}}
  #pragma acc data default(none)
  while(0);

  // expected-warning@+2{{OpenACC construct 'loop' not yet implemented}}
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
  #pragma acc loop default(none)
  while(0);

  // expected-warning@+2{{OpenACC construct 'wait' not yet implemented}}
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'wait' directive}}
  #pragma acc wait default(none)
  while(0);

  // expected-error@+2{{OpenACC 'default' clause is not valid on 'loop' directive}}
  // expected-warning@+1{{OpenACC construct 'loop' not yet implemented}}
#pragma acc loop default(present)
  for(;;);
}
