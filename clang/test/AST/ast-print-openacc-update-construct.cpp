// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s
void uses(bool cond) {
  int I;
  int *iPtr;
  int array[5];
  // CHECK: #pragma acc update self(I)
#pragma acc update self(I)

// CHECK: #pragma acc update self(I) if_present
#pragma acc update self(I) if_present
// CHECK: #pragma acc update self(I) if(cond)
#pragma acc update self(I) if(cond)

// CHECK: #pragma acc update self(I) async
#pragma acc update self(I) async
// CHECK: #pragma acc update self(I) async(*iPtr)
#pragma acc update self(I) async(*iPtr)
// CHECK: #pragma acc update self(I) async(I)
#pragma acc update self(I) async(I)

// CHECK: #pragma acc update self(I) wait(*iPtr, I) async
#pragma acc update self(I) wait(*iPtr, I) async

// CHECK: #pragma acc update self(I) wait(queues: *iPtr, I) async(*iPtr)
#pragma acc update self(I) wait(queues:*iPtr, I) async(*iPtr)

// CHECK: #pragma acc update self(I) wait(devnum: I : *iPtr, I) async(I)
#pragma acc update self(I) wait(devnum:I:*iPtr, I) async(I)

// CHECK: #pragma acc update self(I) wait(devnum: I : queues: *iPtr, I) if(I == array[I]) async(I)
#pragma acc update self(I) wait(devnum:I:queues:*iPtr, I) if(I == array[I]) async(I)

// CHECK: #pragma acc update self(I) device_type(nvidia) dtype(host)
#pragma acc update self(I) device_type(nvidia) dtype(host)

// CHECK: #pragma acc update self(I) device_type(radeon) dtype(multicore)
#pragma acc update self(I) device_type(radeon) dtype(multicore)

// CHECK: #pragma acc update self(I, iPtr, array, array[1], array[1:2])
#pragma acc update self(I, iPtr, array, array[1], array[1:2])

// CHECK: #pragma acc update host(I, iPtr, array, array[1], array[1:2])
#pragma acc update host (I, iPtr, array, array[1], array[1:2])

// CHECK: #pragma acc update device(I, iPtr, array, array[1], array[1:2])
#pragma acc update device(I, iPtr, array, array[1], array[1:2])
}
