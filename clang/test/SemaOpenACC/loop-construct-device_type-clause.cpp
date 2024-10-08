// RUN: %clang_cc1 %s -fopenacc -verify

template<typename T>
void TemplUses() {
#pragma acc loop device_type(I)
  for(;;);
#pragma acc loop dtype(*)
  for(;;);
#pragma acc loop device_type(class)
  for(;;);
#pragma acc loop device_type(private)
  for(;;);
#pragma acc loop device_type(bool)
  for(;;);
#pragma acc kernels dtype(true) device_type(false)
  for(;;);
  // expected-error@+2{{expected ','}}
  // expected-error@+1{{expected identifier}}
#pragma acc loop device_type(T::value)
  for(;;);
}

void Inst() {
  TemplUses<int>(); // #INST
}
