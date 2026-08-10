// RUN: %clang_cc1 %s -fopenacc -verify

void Compute() {
  // expected-error@+1{{expected expression}}
#pragma acc parallel wait()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial wait()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels wait()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel num_gangs()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial num_gangs()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels num_gangs()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel num_workers()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial num_workers()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels num_workers()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel vector_length()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial vector_length()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels vector_length()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel reduction(+:)
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial reduction(+:)
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel copy()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial copy()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels copy()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel copyin()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial copyin(readonly:)
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels copyin()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel copyout()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial copyout(zero:)
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels copyout()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel create()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial create(zero:)
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels create()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel no_create()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial no_create()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels no_create()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel present()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial present()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels present()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel deviceptr()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial deviceptr()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels deviceptr()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel attach()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial attach()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc kernels attach()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel private()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial private()
  ;

  // expected-error@+1{{expected expression}}
#pragma acc parallel firstprivate()
  ;
  // expected-error@+1{{expected expression}}
#pragma acc serial firstprivate()
  ;

  // expected-error@+1{{expected identifier}}
#pragma acc parallel device_type()
  ;
  // expected-error@+1{{expected identifier}}
#pragma acc serial device_type()
  ;
  // expected-error@+1{{expected identifier}}
#pragma acc kernels device_type()
  ;
}

void Data(int i) {
  // expected-error@+1{{expected expression}}
#pragma acc data default(none) wait()
  // expected-error@+1{{expected expression}}
#pragma acc enter data copyin(i) wait()
  // expected-error@+1{{expected expression}}
#pragma acc exit data copyout(i) wait()

  // expected-error@+1{{expected identifier}}
#pragma acc data default(none) device_type()

  // expected-error@+1{{expected expression}}
#pragma acc data copy()

  // expected-error@+1{{expected expression}}
#pragma acc data copyin()
  // expected-error@+1{{expected expression}}
#pragma acc enter data copyin()

  // expected-error@+1{{expected expression}}
#pragma acc data copyout()
  // expected-error@+1{{expected expression}}
#pragma acc exit data copyout()

  // expected-error@+1{{expected expression}}
#pragma acc exit data delete()

  // expected-error@+1{{expected expression}}
#pragma acc exit data detach()

  // expected-error@+1{{expected expression}}
#pragma acc data create()
  // expected-error@+1{{expected expression}}
#pragma acc enter data create()

  // expected-error@+1{{expected expression}}
#pragma acc data default(none) no_create()

  // expected-error@+1{{expected expression}}
#pragma acc data present()

  // expected-error@+1{{expected expression}}
#pragma acc data deviceptr()

  // expected-error@+1{{expected expression}}
#pragma acc data attach()
  // expected-error@+1{{expected expression}}
#pragma acc enter data attach()

  // expected-error@+1{{expected expression}}
#pragma acc host_data use_device()
  ;
}

void Executable(int i) {
  // expected-error@+1{{expected identifier}}
#pragma acc init device_type()
  // expected-error@+1{{expected identifier}}
#pragma acc shutdown device_type()
  // expected-error@+1{{expected identifier}}
#pragma acc set device_num(i) device_type()
  // expected-error@+1{{expected identifier}}
#pragma acc update self(i) device_type()

  // expected-error@+1{{expected expression}}
#pragma acc update self(i) wait()
  // expected-error@+1{{expected expression}}
#pragma acc update self()
  // expected-error@+1{{expected expression}}
#pragma acc update host()
  // expected-error@+1{{expected expression}}
#pragma acc update device()

  // expected-error@+1{{expected expression}}
#pragma acc wait()
}

void Other() {
  // expected-error@+1{{expected expression}}
#pragma acc loop gang()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop worker()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop vector()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop tile()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected identifier}}
#pragma acc loop device_type()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop private()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc loop reduction(+:)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc cache()

  // expected-error@+1{{expected expression}}
#pragma acc declare copy()
  // expected-error@+1{{expected expression}}
#pragma acc declare copyin()
  // expected-error@+1{{expected expression}}
#pragma acc declare copyout()
  // expected-error@+1{{expected expression}}
#pragma acc declare create()
  // expected-error@+1{{expected expression}}
#pragma acc declare present()
  // expected-error@+1{{expected expression}}
#pragma acc declare deviceptr()
  // expected-error@+1{{expected expression}}
#pragma acc declare device_resident()
  // expected-error@+1{{expected expression}}
#pragma acc declare link()

  auto L1 =[]{};

  // expected-error@+1{{expected identifier}}
#pragma acc routine(L1) seq device_type()

  // expected-error@+1{{expected identifier}}
#pragma acc routine seq device_type()
  auto L2 =[]{};
}

void Combined() {
  // expected-error@+1{{expected expression}}
#pragma acc parallel loop gang()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop gang()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop gang()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop tile()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop tile()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop tile()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected identifier}}
#pragma acc parallel loop device_type()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected identifier}}
#pragma acc serial loop device_type()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected identifier}}
#pragma acc kernels loop device_type()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop reduction(+:)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop reduction(+:)
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop reduction(+:)
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop wait()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop wait()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop wait()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop num_gangs()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop num_gangs()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop copy()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop copy()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop copy()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop copyin()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop copyin()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop copyin()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop copyout()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop copyout()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop copyout()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop create()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop create()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop create()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop no_create()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop no_create()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop present()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop present()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop present()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop deviceptr()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop deviceptr()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop deviceptr()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop attach()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop attach()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop attach()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop private()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop private()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc kernels loop private()
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{expected expression}}
#pragma acc parallel loop firstprivate()
  for(int i = 0; i < 5; ++i);
  // expected-error@+1{{expected expression}}
#pragma acc serial loop firstprivate()
  for(int i = 0; i < 5; ++i);
}
