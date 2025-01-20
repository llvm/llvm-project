// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

void uses() {
  int *iPtr;
  int I;
  float array[5];

// CHECK: #pragma acc wait() if(I == array[I])
#pragma acc wait() if(I == array[I])

// CHECK: #pragma acc wait(*iPtr, I) async
#pragma acc wait(*iPtr, I) async

// CHECK: #pragma acc wait(queues: *iPtr, I) async(*iPtr)
#pragma acc wait(queues:*iPtr, I) async(*iPtr)

// CHECK: #pragma acc wait(devnum: I : *iPtr, I) async(I)
#pragma acc wait(devnum:I:*iPtr, I) async(I)

// CHECK: #pragma acc wait(devnum: I : queues: *iPtr, I) if(I == array[I]) async(I)
#pragma acc wait(devnum:I:queues:*iPtr, I) if(I == array[I]) async(I)
}
