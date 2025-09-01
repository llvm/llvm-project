// RUN: %clang_cc1 %s -fopenacc -verify

void Test() {
  int I;
  struct NotConvertible{} NC;
  // No special rules for this clause on the data constructs, so not much to
  // test that isn't covered by combined/compute.
#pragma acc data copyin(I) async(I)
  ;
#pragma acc enter data copyin(I) async(I)
#pragma acc exit data copyout(I) async(I)
  // expected-error@+1{{OpenACC 'async' clause is not valid on 'host_data' directive}}
#pragma acc host_data use_device(I) async(I)
  ;

  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc data copyin(NC) async(NC)
  ;
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc enter data copyin(NC) async(NC)
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc exit data copyout(NC) async(NC)
  // expected-error@+1{{OpenACC clause 'async' requires expression of integer type ('struct NotConvertible' invalid)}}
#pragma acc host_data use_device(NC) async(NC)
  ;

  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous 'async' clause is here}}
#pragma acc data copyin(I) async(I) async(I)
  ;
  // expected-error@+2{{expected ')'}}
  // expected-note@+1{{to match this '('}}
#pragma acc enter data copyin(I) async(I, I)
  //
  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous 'async' clause is here}}
#pragma acc data default(none) async async
  while(1);

  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous 'async' clause is here}}
#pragma acc data default(none) async(1) async(2)
  while(1);

  // expected-error@+2{{OpenACC 'async' clause cannot appear more than once on a 'data' directive}}
  // expected-note@+1{{previous 'async' clause is here}}
#pragma acc data default(none) async(1) async(2)
  while(1);

  // expected-error@+3{{OpenACC 'async' clause cannot appear more than once in a 'device_type' region on a 'data' directive}}
  // expected-note@+2{{previous 'async' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) async(1) device_type(*) async(1) async(2)
  while(1);
  // expected-error@+3{{OpenACC 'async' clause cannot appear more than once in a 'device_type' region on a 'data' directive}}
  // expected-note@+2{{previous 'async' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) async device_type(*) async async
  while(1);
  // expected-error@+3{{OpenACC 'async' clause cannot appear more than once in a 'device_type' region on a 'data' directive}}
  // expected-note@+2{{previous 'async' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) async(1) device_type(*) async async(2)
  while(1);

  // expected-error@+3{{OpenACC 'async' clause cannot appear more than once in a 'device_type' region on a 'data' directive}}
  // expected-note@+2{{previous 'async' clause is here}}
  // expected-note@+1{{active 'device_type' clause here}}
#pragma acc data default(none) device_type(*) async async
  while(1);
}
