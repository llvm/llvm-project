// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void TemplUses() {
#pragma acc parallel loop device_type(default)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop dtype(*)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop device_type(nvidia)
  for(int i = 0; i < 5; ++i);
#pragma acc parallel loop device_type(radeon)
  for(int i = 0; i < 5; ++i);
#pragma acc serial loop device_type(host)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels loop dtype(multicore) device_type(host)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop device_type(T::value)
  for(int i = 0; i < 5; ++i);
}

void Inst() {
  TemplUses<int>(); // #INST
}
