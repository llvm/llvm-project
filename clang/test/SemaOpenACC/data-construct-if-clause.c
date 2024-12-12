// RUN: %clang_cc1 %s -fopenacc -verify

void Foo() {
  int Var;
#pragma acc data default(present) if(1)
  ;
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(present) if(1) if (2)
  ;

  // expected-warning@+1{{OpenACC clause 'copyin' not yet implemented}}
#pragma acc enter data copyin(Var) if(1)

  // expected-warning@+3{{OpenACC clause 'copyin' not yet implemented}}
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'enter data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc enter data copyin(Var) if(1) if (2)

  // expected-warning@+1{{OpenACC clause 'copyout' not yet implemented}}
#pragma acc exit data copyout(Var) if(1)
  // expected-warning@+3{{OpenACC clause 'copyout' not yet implemented}}
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'exit data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc exit data copyout(Var) if(1) if (2)

  // expected-warning@+1{{OpenACC clause 'use_device' not yet implemented}}
#pragma acc host_data use_device(Var) if(1)
  ;
  // expected-warning@+3{{OpenACC clause 'use_device' not yet implemented}}
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'host_data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc host_data use_device(Var) if(1) if (2)
  ;
}
