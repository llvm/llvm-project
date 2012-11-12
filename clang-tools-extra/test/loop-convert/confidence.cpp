// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: loop-convert . %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: loop-convert . %t.cpp -A2 -- -I %S/Inputs
// RUN: FileCheck -check-prefix=RISKY -input-file=%t.cpp %s

#include "structures.h"

void f() {
  const int N = 5;
  const int M = 7;
  int (*pArr)[N];
  int Arr[N][M];
  int sum = 0;

  for (int i = 0; i < M; ++i) {
    sum += Arr[0][i];
  }
  // CHECK: for (int i = 0; i < M; ++i) {
  // CHECK-NEXT: sum += Arr[0][i];
  // CHECK-NEXT: }
  // RISKY: for (auto & [[VAR:[a-z_]+]] : Arr[0]) {
  // RISKY-NEXT: sum += [[VAR]];
  // RISKY-NEXT: }

  for (int i = 0; i < N; ++i) {
    sum += (*pArr)[i];
  }
  // RISKY: for (auto & [[VAR:[a-z_]+]] : *pArr) {
  // RISKY-NEXT: sum += [[VAR]];
  // RISKY-NEXT: }
  // CHECK: for (int i = 0; i < N; ++i) {
  // CHECK-NEXT: sum += (*pArr)[i];
  // CHECK-NEXT: }
}
