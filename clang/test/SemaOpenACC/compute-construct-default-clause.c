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

  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'parallel' directive}}
  // expected-note@+1{{previous clause is here}}
  #pragma acc parallel self default(present) private(i) default(none) copy(i)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected '('}}
  #pragma acc serial self default private(i) default(none) if(i)
  for(int i = 0; i < 5; ++i);

  #pragma acc kernels default(none)
  for(int i = 0; i < 5; ++i);

  #pragma acc data default(none)
  while(0);

  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
  #pragma acc loop default(none)
  for(int i = 5; i < 10;++i);

  // expected-error@+1{{OpenACC 'default' clause is not valid on 'wait' directive}}
  #pragma acc wait default(none)
  while(0);

  // expected-error@+1{{OpenACC 'default' clause is not valid on 'loop' directive}}
#pragma acc loop default(present)
  for(int i = 5; i < 10;++i);
}
