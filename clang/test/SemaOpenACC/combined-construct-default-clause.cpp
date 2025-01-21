// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void SingleOnly() {
  #pragma acc parallel loop default(none)
  for (unsigned I = 0; I < 5; ++I);

  int i;

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'parallel loop' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc parallel loop default(present) async default(none)
  for (unsigned I = 0; I < 5; ++I);

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'serial loop' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc serial loop async default(present) copy(i) default(none) self
  for (unsigned I = 0; I < 5; ++I);

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'kernels loop' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc kernels loop async default(present) copy(i) default(none) self
  for (unsigned I = 0; I < 5; ++I);

  // expected-error@+1{{expected '('}}
  #pragma acc parallel loop async default(none) copy(i) default self
  for (unsigned I = 0; I < 5; ++I);
}

void Instantiate() {
  SingleOnly<int>();
}
