// RUN: %clang_cc1 %s -fopenacc -verify

void Foo() {
  int Var;
#pragma acc data default(present) if(1)
  ;
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data default(present) if(1) if (2)
  ;

#pragma acc enter data copyin(Var) if(1)

  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'enter data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc enter data copyin(Var) if(1) if (2)

#pragma acc exit data copyout(Var) if(1)
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'exit data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc exit data copyout(Var) if(1) if (2)

#pragma acc host_data use_device(Var) if(1)
  ;
  // expected-error@+2{{OpenACC 'if' clause cannot appear more than once on a 'host_data' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc host_data use_device(Var) if(1) if (2)
  ;
}
