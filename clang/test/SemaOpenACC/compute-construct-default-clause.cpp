// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void SingleOnly() {
  #pragma acc parallel default(none)
  while(false);

  int i;

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc parallel default(present) async default(none)
  while(false);

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'serial' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc serial async default(present) copy(i) default(none) self
  while(false);

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'kernels' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc kernels async default(present) copy(i) default(none) self
  while(false);

  // expected-error@+1{{expected '('}}
  #pragma acc parallel async default(none) copy(i) default self
  while(false);
}

void Instantiate() {
  SingleOnly<int>();
}
