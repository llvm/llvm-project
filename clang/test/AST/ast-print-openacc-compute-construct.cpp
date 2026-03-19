// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

void foo() {
  int i;
  int *iPtr;
  float array[5];
  float *arrayPtr[5];
// CHECK: #pragma acc parallel default(none)
// CHECK-NEXT: while (true)
#pragma acc parallel default(none)
  while(true);
// CHECK: #pragma acc serial default(present)
// CHECK-NEXT: while (true)
#pragma acc serial default(present)
  while(true);
// CHECK: #pragma acc kernels if(i == array[1])
// CHECK-NEXT: while (true)
#pragma acc kernels if(i == array[1])
  while(true);
// CHECK: #pragma acc parallel self(i == 3)
// CHECK-NEXT: while (true)
#pragma acc parallel self(i == 3)
  while(true);

// CHECK: #pragma acc parallel num_gangs(i, (int)array[2])
// CHECK-NEXT: while (true)
#pragma acc parallel num_gangs(i, (int)array[2])
  while(true);

// CHECK: #pragma acc parallel num_workers(i)
// CHECK-NEXT: while (true)
#pragma acc parallel num_workers(i)
  while(true);

// CHECK: #pragma acc parallel vector_length((int)array[1])
// CHECK-NEXT: while (true)
#pragma acc parallel vector_length((int)array[1])
  while(true);

// CHECK: #pragma acc parallel private(i, array[1], array, array[1:2])
#pragma acc parallel private(i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel firstprivate(i, array[1], array, array[1:2])
#pragma acc parallel firstprivate(i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel no_create(i, array[1], array, array[1:2])
#pragma acc parallel no_create(i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel present(i, array[1], array, array[1:2])
#pragma acc parallel present(i, array[1], array, array[1:2])
  while(true);
// CHECK: #pragma acc parallel no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
#pragma acc parallel no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel copy(i, array[1], array, array[1:2]) pcopy(alwaysin: i, array[1], array, array[1:2]) present_or_copy(always, alwaysout: i, array[1], array, array[1:2])
#pragma acc parallel copy(i, array[1], array, array[1:2]) pcopy(alwaysin:i, array[1], array, array[1:2]) present_or_copy(always, alwaysout: i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel copyin(i, array[1], array, array[1:2]) pcopyin(readonly: i, array[1], array, array[1:2]) present_or_copyin(always, readonly: i, array[1], array, array[1:2])
#pragma acc parallel copyin(i, array[1], array, array[1:2]) pcopyin(readonly:i, array[1], array, array[1:2]) present_or_copyin(readonly, always: i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(always, zero: i, array[1], array, array[1:2])
#pragma acc parallel copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(always, zero: i, array[1], array, array[1:2])
  while(true);

// CHECK: #pragma acc parallel create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])
#pragma acc parallel create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])
  while(true);

  // CHECK: #pragma acc serial attach(iPtr, arrayPtr[0])
#pragma acc serial attach(iPtr, arrayPtr[0])
  while(true);

  // CHECK: #pragma acc kernels deviceptr(iPtr, arrayPtr[0])
#pragma acc kernels deviceptr(iPtr, arrayPtr[0])
  while(true);

  // CHECK: #pragma acc kernels async(*iPtr)
#pragma acc kernels async(*iPtr)
  while(true);

  // CHECK: #pragma acc kernels async
#pragma acc kernels async
  while(true);

// CHECK: #pragma acc parallel wait
#pragma acc parallel wait
  while(true);

// CHECK: #pragma acc parallel wait
#pragma acc parallel wait
  while(true);

// CHECK: #pragma acc parallel wait(*iPtr, i)
#pragma acc parallel wait(*iPtr, i)
  while(true);

// CHECK: #pragma acc parallel wait(queues: *iPtr, i)
#pragma acc parallel wait(queues:*iPtr, i)
  while(true);

// CHECK: #pragma acc parallel wait(devnum: i : *iPtr, i)
#pragma acc parallel wait(devnum:i:*iPtr, i)
  while(true);

// CHECK: #pragma acc parallel wait(devnum: i : queues: *iPtr, i)
#pragma acc parallel wait(devnum:i:queues:*iPtr, i)
  while(true);

  bool SomeB;
  struct SomeStruct{} SomeStructImpl;

//CHECK: #pragma acc parallel dtype(default)
#pragma acc parallel dtype(default)
  while(true);

//CHECK: #pragma acc parallel device_type(radeon)
#pragma acc parallel device_type(radeon)
  while(true);

//CHECK: #pragma acc parallel device_type(nvidia)
#pragma acc parallel device_type(nvidia)
  while(true);

//CHECK: #pragma acc parallel dtype(multicore)
#pragma acc parallel dtype(multicore)
  while(true);

//CHECK: #pragma acc parallel device_type(host)
#pragma acc parallel device_type (host)
  while(true);

//CHECK: #pragma acc parallel reduction(*: i)
#pragma acc parallel reduction(*: i)
  while(true);
//CHECK: #pragma acc parallel reduction(max: SomeB)
#pragma acc parallel reduction(max: SomeB)
  while(true);
//CHECK: #pragma acc parallel reduction(&: i)
#pragma acc parallel reduction(&: i)
  while(true);
//CHECK: #pragma acc parallel reduction(|: SomeB)
#pragma acc parallel reduction(|: SomeB)
  while(true);
//CHECK: #pragma acc parallel reduction(&&: i)
#pragma acc parallel reduction(&&: i)
  while(true);
//CHECK: #pragma acc parallel reduction(||: SomeB)
#pragma acc parallel reduction(||: SomeB)
  while(true);
}

