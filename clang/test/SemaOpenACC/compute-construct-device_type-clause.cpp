// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void TemplUses() {
#pragma acc parallel device_type(I)
  while(true);
#pragma acc parallel dtype(*)
  while(true);
#pragma acc parallel device_type(class)
  while(true);
#pragma acc parallel device_type(private)
  while(true);
#pragma acc parallel device_type(bool)
  while(true);
#pragma acc kernels dtype(true) device_type(false)
  while(true);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type(T::value)
  while(true);
}

void Inst() {
  TemplUses<int>(); // #INST
}
