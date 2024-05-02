// RUN: %clang_cc1 -fopenacc -ast-print %s -o - | FileCheck %s

void foo() {
  int i;
  float array[5];
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
}

