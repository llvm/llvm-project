// RUN: %clang_cc1 %s -fopenacc -verify

void uses() {
  int Var;
#pragma acc data device_type(foo) async
  ;
#pragma acc data device_type(foo) wait
  ;
#pragma acc data device_type(foo) dtype(false)
  ;
#pragma acc data dtype(foo) device_type(false)
  ;

  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) if(1)
  ;
  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) copy(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) copyin(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) copyout(Var)
  ;
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) no_create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) present(Var)
  ;
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) deviceptr(Var)
  ;
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) attach(Var)
  ;
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{previous clause is here}}
#pragma acc data device_type(foo) default(none)
  ;
}
