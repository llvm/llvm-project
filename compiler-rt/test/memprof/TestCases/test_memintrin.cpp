// Check profile with calls to memory intrinsics.

// RUN: %clangxx_memprof -O0 %s -o %t && %run %t 2>&1 | FileCheck %s

// This is actually:
//  Memory allocation stack id = STACKIDP
//   alloc_count 1, size (ave/min/max) 40.00 / 40 / 40
//   access_count (ave/min/max): 3.00 / 3 / 3
// but we need to look for them in the same CHECK to get the correct STACKIDP.
// CHECK-DAG:  Memory allocation stack id = [[STACKIDP:[0-9]+]]{{[[:space:]].*}} alloc_count 1, size (ave/min/max) 40.00 / 40 / 40{{[[:space:]].*}} access_count (ave/min/max): 3.00 / 3 / 3
//
// This is actually:
//  Memory allocation stack id = STACKIDQ
//   alloc_count 1, size (ave/min/max) 20.00 / 20 / 20
//   access_count (ave/min/max): 2.00 / 2 / 2
// but we need to look for them in the same CHECK to get the correct STACKIDQ.
// CHECK-DAG:  Memory allocation stack id = [[STACKIDQ:[0-9]+]]{{[[:space:]].*}} alloc_count 1, size (ave/min/max) 20.00 / 20 / 20{{[[:space:]].*}} access_count (ave/min/max): 2.00 / 2 / 2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
  // This is actually:
  //  Stack for id STACKIDP:
  //    #0 {{.*}} in operator new
  //    #1 {{.*}} in main {{.*}}:@LINE+1
  //  but we need to look for them in the same CHECK-DAG.
  // CHECK-DAG: Stack for id [[STACKIDP]]:{{[[:space:]].*}} #0 {{.*}} in operator new{{.*[[:space:]].*}} #1 {{.*}} in main {{.*}}:[[@LINE+1]]
  int *p = new int[10];

  // This is actually:
  //  Stack for id STACKIDQ:
  //    #0 {{.*}} in operator new
  //    #1 {{.*}} in main {{.*}}:@LINE+1
  //  but we need to look for them in the same CHECK-DAG.
  // CHECK-DAG: Stack for id [[STACKIDQ]]:{{[[:space:]].*}} #0 {{.*}} in operator new{{.*[[:space:]].*}} #1 {{.*}} in main {{.*}}:[[@LINE+1]]
  int *q = new int[5];

  memset(p, 1, 10);
  memcpy(q, p, 5);
  int x = memcmp(p, q, 5);

  delete p;
  delete q;

  return x;
}
