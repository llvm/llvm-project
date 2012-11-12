// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: loop-convert . %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

void f() {
  const int N = 10;
  const int M = 15;
  Val Arr[N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int k = Arr[i].x + Arr[j].x;
      // The repeat is there to allow FileCheck to make sure the two variable
      // names aren't the same.
      int l = Arr[i].x + Arr[j].x;
    }
  }
  // CHECK: for (auto & [[VAR:[a-zA-Z_]+]] : Arr)
  // CHECK-NEXT: for (auto & [[INNERVAR:[a-zA-Z_]+]] : Arr)
  // CHECK-NEXT: int k = [[VAR]].x + [[INNERVAR]].x;
  // CHECK-NOT: int l = [[VAR]].x + [[VAR]].x;

  Val Nest[N][M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. Update this test when that changes!
  // CHECK: for (auto & [[VAR:[a-zA-Z_]+]] : Nest)
  // CHECK-NEXT: for (int j = 0; j < M; ++j)
  // CHECK-NEXT: printf("Got item %d", [[VAR]][j].x);

  // Note that the order of M and N are switched for this test.
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // CHECK-NOT: for (auto & {{[a-zA-Z_]+}} : Nest[i])
  // CHECK: for (int j = 0; j < M; ++j)
  // CHECK-NEXT: for (auto & [[VAR:[a-zA-Z_]+]] : Nest)
  // CHECK-NEXT: printf("Got item %d", [[VAR]][j].x);
  Nested<T> NestT;
  for (Nested<T>::iterator I = NestT.begin(), E = NestT.end(); I != E; ++I) {
    for (T::iterator TI = (*I).begin(), TE = (*I).end(); TI != TE; ++TI) {
      printf("%d", *TI);
    }
  }
  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. Update this test when that changes!
  // CHECK: for (auto & [[VAR:[a-zA-Z_]+]] : NestT) {
  // CHECK-NEXT: for (T::iterator TI = ([[VAR]]).begin(), TE = ([[VAR]]).end(); TI != TE; ++TI) {
  // CHECK-NEXT: printf("%d", *TI);
}
