// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void TemplUses() {
#pragma acc loop device_type(host)
  for(int i = 0; i < 5; ++i);
#pragma acc loop dtype(*)
  for(int i = 0; i < 5; ++i);
#pragma acc loop device_type(default)
  for(int i = 0; i < 5; ++i);
#pragma acc loop device_type(multicore)
  for(int i = 0; i < 5; ++i);
#pragma acc loop device_type(host)
  for(int i = 0; i < 5; ++i);
#pragma acc kernels dtype(radeon) device_type(nvidia)
  for(int i = 0; i < 5; ++i);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc loop device_type(T::value)
  for(int i = 0; i < 5; ++i);
}

void Inst() {
  TemplUses<int>(); // #INST
}
