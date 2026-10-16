// RUN: %clang_cc1 %s -fopenacc -verify 

void foo(int i) {
#pragma acc parallel device_type(*)
  ;
#pragma acc serial device_type(default)
  ;
#pragma acc kernels device_type(Default)
  ;
#pragma acc data copy(i) device_type(nvidia)
  ;
#pragma acc init device_type(acc_device_nvidia)
  ;
#pragma acc shutdown device_type(radeon)
  ;
#pragma acc set device_type(host)
  ;
#pragma acc update self(i) device_type(multicore)
  ;
#pragma acc loop device_type(DEFAULT)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc parallel loop device_type(HOST)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc serial loop device_type(MuLtIcOrE)
  for(unsigned I = 0; I < 5; ++I);
#pragma acc kernels loop device_type(radEon)
  for(unsigned I = 0; I < 5; ++I);
}

#pragma acc routine(foo) seq device_type(host)

#pragma acc routine seq device_type(acc_device_nvidia)
void bar(){}

// Anything outside of the above list is invalid.
void invalid() {
  // expected-error@+1{{invalid value 'other' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc parallel device_type(other)
  ;
  // expected-error@+1{{invalid value 'invalid' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc kernels device_type(invalid)
  ;
  // expected-error@+1{{invalid value 'invalid' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc kernels device_type(invalid, nvidia)
  ;
  // expected-error@+1{{invalid value 'invalid' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc kernels device_type(nvidia, invalid)
  ;
  // expected-error@+1{{invalid value 'invalid' in 'dtype' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc kernels dtype(nvidia, invalid, radeon)
  ;
  // expected-error@+2{{invalid value 'invalid' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
  // expected-error@+1{{invalid value 'invalid2' in 'device_type' clause; valid values are 'default', 'nvidia', 'acc_device_nvidia', 'radeon', 'host', 'multicore'}}
#pragma acc kernels device_type(nvidia, invalid, radeon, invalid2)
  ;
}
