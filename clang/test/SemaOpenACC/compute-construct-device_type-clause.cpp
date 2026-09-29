// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void TemplUses() {
#pragma acc parallel device_type(default)
  while(true);
#pragma acc parallel dtype(*)
  while(true);
#pragma acc parallel device_type(nvidia)
  while(true);
#pragma acc parallel device_type(radeon)
  while(true);
#pragma acc parallel device_type(host)
  while(true);
#pragma acc kernels dtype(multicore) device_type(host)
  while(true);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type(T::value)
  while(true);
}

void Inst() {
  TemplUses<int>(); // #INST
}
