// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s
void uses(bool cond) {
  int I;
  int *iPtr;
  int array[5];
  // CHECK: #pragma acc update
#pragma acc update

// CHECK: #pragma acc update if_present
#pragma acc update if_present
// CHECK: #pragma acc update if(cond)
#pragma acc update if(cond)

// CHECK: #pragma acc update async
#pragma acc update async
// CHECK: #pragma acc update async(*iPtr)
#pragma acc update async(*iPtr)
// CHECK: #pragma acc update async(I)
#pragma acc update async(I)

// CHECK: #pragma acc update wait(*iPtr, I) async
#pragma acc update wait(*iPtr, I) async

// CHECK: #pragma acc update wait(queues: *iPtr, I) async(*iPtr)
#pragma acc update wait(queues:*iPtr, I) async(*iPtr)

// CHECK: #pragma acc update wait(devnum: I : *iPtr, I) async(I)
#pragma acc update wait(devnum:I:*iPtr, I) async(I)

// CHECK: #pragma acc update wait(devnum: I : queues: *iPtr, I) if(I == array[I]) async(I)
#pragma acc update wait(devnum:I:queues:*iPtr, I) if(I == array[I]) async(I)

// CHECK: #pragma acc update device_type(I) dtype(H)
#pragma acc update device_type(I) dtype(H)

// CHECK: #pragma acc update device_type(J) dtype(K)
#pragma acc update device_type(J) dtype(K)

// CHECK: #pragma acc update self(I, iPtr, array, array[1], array[1:2])
#pragma acc update self(I, iPtr, array, array[1], array[1:2])
}
