// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void SingleOnly() {
  #pragma acc parallel default(none)
  while(false);

  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc parallel default(present) seq default(none)
  while(false);

  // expected-warning@+5{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'serial' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc serial seq default(present) seq default(none) seq
  while(false);

  // expected-warning@+5{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+4{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc kernels seq default(present) seq default(none) seq
  while(false);

  // expected-warning@+3{{OpenACC clause 'seq' not yet implemented}}
  // expected-warning@+2{{OpenACC clause 'seq' not yet implemented}}
  // expected-error@+1{{expected '('}}
  #pragma acc parallel seq default(none) seq default seq
  while(false);
}

void Instantiate() {
  SingleOnly<int>();
}
