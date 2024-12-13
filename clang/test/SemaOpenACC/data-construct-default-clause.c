// RUN: %clang_cc1 %s -fopenacc -verify

void use() {
  // expected-error@+1{{invalid value for 'default' clause; expected 'present' or 'none'}}
#pragma acc data default(garbage)
  ;
#pragma acc data default(present)
  ;
#pragma acc data default(none)
  ;
  // expected-error@+2{{OpenACC 'default' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(none) default(present)
  ;
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'enter data' directive}}
#pragma acc enter data default(present)
  ;
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'exit data' directive}}
#pragma acc exit data default(none)
  ;
  // expected-error@+1{{OpenACC 'default' clause is not valid on 'host_data' directive}}
#pragma acc host_data default(present)
  ;
}
