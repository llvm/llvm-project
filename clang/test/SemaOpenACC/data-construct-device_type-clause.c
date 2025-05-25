// RUN: %clang_cc1 %s -fopenacc -verify

void uses() {
  int Var;
#pragma acc data default(none) device_type(radeon) async
  ;
#pragma acc data default(none) device_type(radeon) wait
  ;
#pragma acc data default(none) device_type(radeon) dtype(nvidia)
  ;
#pragma acc data default(none) dtype(radeon) device_type(nvidia)
  ;

  // expected-error@+2{{OpenACC clause 'if' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) if(1)
  ;
  // expected-error@+2{{OpenACC clause 'copy' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) copy(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyin' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) copyin(Var)
  ;
  // expected-error@+2{{OpenACC clause 'copyout' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) copyout(Var)
  ;
  // expected-error@+2{{OpenACC clause 'create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'no_create' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) no_create(Var)
  ;
  // expected-error@+2{{OpenACC clause 'present' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) present(Var)
  ;
  // expected-error@+2{{OpenACC clause 'deviceptr' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) deviceptr(Var)
  ;
  // expected-error@+2{{OpenACC clause 'attach' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(radeon) attach(Var)
  ;
  // expected-error@+3{{OpenACC 'data' construct must have at least one 'attach', 'copy', 'copyin', 'copyout', 'create', 'default', 'deviceptr', 'no_create', or 'present' clause}}
  // expected-error@+2{{OpenACC clause 'default' may not follow a 'device_type' clause in a 'data' construct}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data device_type(radeon) default(none)
  ;
}
