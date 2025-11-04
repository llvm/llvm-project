// RUN: %clang_cc1 -fopenacc -Wno-openacc-deprecated-clause-alias -ast-print %s -o - | FileCheck %s

constexpr int get_value() { return 1; }
void foo() {
  int *iPtr;
// CHECK: #pragma acc parallel loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc parallel loop auto
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop auto
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop seq
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop seq
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop independent
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop independent
  for(int i = 0;i<5;++i);

  bool SomeB;
  struct SomeStruct{} SomeStructImpl;

//CHECK: #pragma acc parallel loop dtype(default)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop dtype(default)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc serial loop device_type(radeon, host)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop device_type(radeon, host)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc kernels loop device_type(nvidia)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop device_type(nvidia)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc parallel loop dtype(multicore)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop dtype(multicore)
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc serial loop device_type(default)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop device_type (default)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels loop dtype(host)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop dtype(host)
  for(int i = 0;i<5;++i);

  int i;
  float array[5];

// CHECK: #pragma acc parallel loop self(i == 3)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop self(i == 3)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels loop if(i == array[1])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop if(i == array[1])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop default(none)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop default(none)
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop default(present)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop default(present)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop private(i, array[1], array, array[1:2])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop private(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial loop firstprivate(i, array[1], array, array[1:2])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop firstprivate(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

  // CHECK: #pragma acc kernels loop async(*iPtr)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop async(*iPtr)
  for(int i = 0;i<5;++i);

  // CHECK: #pragma acc kernels loop async
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop async
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop present(i, array[1], array, array[1:2])
#pragma acc parallel loop present(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop present(i, array[1], array, array[1:2])
#pragma acc serial loop present(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop present(i, array[1], array, array[1:2])
#pragma acc kernels loop present(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

  float *arrayPtr[5];

  // CHECK: #pragma acc kernels loop deviceptr(iPtr, arrayPtr[0])
#pragma acc kernels loop deviceptr(iPtr, arrayPtr[0])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop wait
#pragma acc parallel loop wait
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop wait(*iPtr, i)
#pragma acc parallel loop wait(*iPtr, i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop wait(queues: *iPtr, i)
#pragma acc parallel loop wait(queues:*iPtr, i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop wait(devnum: i : *iPtr, i)
#pragma acc parallel loop wait(devnum:i:*iPtr, i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop wait(devnum: i : queues: *iPtr, i)
#pragma acc parallel loop wait(devnum:i:queues:*iPtr, i)
  for(int i = 0;i<5;++i);

  // CHECK: #pragma acc serial loop attach(iPtr, arrayPtr[0])
#pragma acc serial loop attach(iPtr, arrayPtr[0])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop no_create(i, array[1], array, array[1:2])
#pragma acc parallel loop no_create(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
#pragma acc parallel loop no_create(i, array[1], array, array[1:2]) present(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop copy(alwaysin: i, array[1], array, array[1:2]) pcopy(i, array[1], array, array[1:2]) present_or_copy(i, array[1], array, array[1:2])
#pragma acc parallel loop copy(alwaysin: i, array[1], array, array[1:2]) pcopy(i, array[1], array, array[1:2]) present_or_copy(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop copyin(i, array[1], array, array[1:2]) pcopyin(readonly: i, array[1], array, array[1:2]) present_or_copyin(always, alwaysin: i, array[1], array, array[1:2])
#pragma acc parallel loop copyin(i, array[1], array, array[1:2]) pcopyin(readonly:i, array[1], array, array[1:2]) present_or_copyin(always, alwaysin: i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(always, alwaysout: i, array[1], array, array[1:2])
#pragma acc parallel loop copyout(i, array[1], array, array[1:2]) pcopyout(zero: i, array[1], array, array[1:2]) present_or_copyout(always, alwaysout: i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])
#pragma acc parallel loop create(i, array[1], array, array[1:2]) pcreate(zero: i, array[1], array, array[1:2]) present_or_create(i, array[1], array, array[1:2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop collapse(1)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop collapse(1)
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc serial loop collapse(force:1)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop collapse(force:1)
  for(int i = 0;i<5;++i);
// CHECK: #pragma acc kernels loop collapse(2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop collapse(2)
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i);
// CHECK: #pragma acc parallel loop collapse(force:2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop collapse(force:2)
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial loop tile(1, 3, *, get_value())
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop tile(1, 3, *, get_value())
  for(int i = 0;i<5;++i)
    for(int i = 0;i<5;++i)
      for(int i = 0;i<5;++i)
        for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop num_gangs(i, (int)array[2])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop num_gangs(i, (int)array[2])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop num_workers(i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop num_workers(i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop vector_length((int)array[1])
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop vector_length((int)array[1])
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(static:i) gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(static: i, dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(static:i, dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(static: i) gang(dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(static:i) gang(dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop gang(static: i, dim: 2)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop gang(static:i, dim:2)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop gang(i) gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc kernels loop gang(num: i) gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop gang(num:i) gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial loop gang(static: i)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop gang(static:i)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc serial loop gang(static: *)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop gang(static:*)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop worker
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc parallel loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop worker
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc serial loop worker
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop worker
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc kernels loop worker(num: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop worker(5)
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc kernels loop worker(num: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop worker(num:5)
  for(int i = 0;i<5;++i);

  // CHECK: #pragma acc parallel loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop vector
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop vector(5)
  for(int i = 0;i<5;++i);

// CHECK: #pragma acc parallel loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc parallel loop vector(length:5)
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc kernels loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop vector
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc kernels loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop vector(5)
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc kernels loop vector(length: 5)
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc kernels loop vector(length:5)
  for(int i = 0;i<5;++i);

// CHECK-NEXT: #pragma acc serial loop vector
// CHECK-NEXT: for (int i = 0; i < 5; ++i)
// CHECK-NEXT: ;
#pragma acc serial loop vector
  for(int i = 0;i<5;++i);

//CHECK: #pragma acc serial loop reduction(*: i)
#pragma acc serial loop reduction(*: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc kernels loop reduction(max: SomeB)
#pragma acc kernels loop reduction(max: SomeB)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc serial loop reduction(&: i)
#pragma acc serial loop reduction(&: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc kernels loop reduction(|: SomeB)
#pragma acc kernels loop reduction(|: SomeB)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc serial loop reduction(&&: i)
#pragma acc serial loop reduction(&&: i)
  for(int i = 0;i<5;++i);
//CHECK: #pragma acc kernels loop reduction(||: SomeB)
#pragma acc kernels loop reduction(||: SomeB)
  for(int i = 0;i<5;++i);
}
