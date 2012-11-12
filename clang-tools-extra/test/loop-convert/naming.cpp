// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: loop-convert . %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

const int N = 10;
int nums[N];
int sum = 0;

Val Arr[N];
Val &func(Val &);

void aliasing() {
  // The extra blank braces are left as a placeholder for after the variable
  // declaration is deleted.
  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i]; { }
    int y = t.x;
  }
  // CHECK: for (auto & t : Arr)
  // CHECK-NEXT: { }
  // CHECK-NEXT: int y = t.x;

  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK: for (auto & [[VAR:[a-z_]+]] : Arr)
  // CHECK-NEXT: Val &t = [[VAR]];
  // CHECK-NEXT: int y = t.x;
  // CHECK-NEXT: int z = [[VAR]].x + t.x;

  for (int i = 0; i < N; ++i) {
    Val t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK: for (auto & [[VAR:[a-z_]+]] : Arr)
  // CHECK-NEXT: Val t = [[VAR]];
  // CHECK-NEXT: int y = t.x;
  // CHECK-NEXT: int z = [[VAR]].x + t.x;

  for (int i = 0; i < N; ++i) {
    Val &t = func(Arr[i]);
    int y = t.x;
  }
  // CHECK: for (auto & [[VAR:[a-z_]+]] : Arr)
  // CHECK-NEXT: Val &t = func([[VAR]]);
  // CHECK-NEXT: int y = t.x;
}

void sameNames() {
  int num = 0;
  for (int i = 0; i < N; ++i) {
    printf("Fibonacci number is %d\n", nums[i]);
    sum += nums[i] + 2 + num;
    (void) nums[i];
  }
  // CHECK: int num = 0;
  // CHECK-NEXT: for (auto & [[VAR:[a-z_]+]] : nums)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", [[VAR]]);
  // CHECK-NEXT: sum += [[VAR]] + 2 + num;
  // CHECK-NOT: (void) num;
  // CHECK: }
}
