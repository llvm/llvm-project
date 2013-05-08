// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: cpp11-migrate -loop-convert %t.cpp -risk=risky -- -I %S/Inputs
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
  // RISKY: for (auto & elem : Arr[0]) {
  // RISKY-NEXT: sum += elem;
  // RISKY-NEXT: }

  for (int i = 0; i < N; ++i) {
    sum += (*pArr)[i];
  }
  // RISKY: for (auto & elem : *pArr) {
  // RISKY-NEXT: sum += elem;
  // RISKY-NEXT: }
  // CHECK: for (int i = 0; i < N; ++i) {
  // CHECK-NEXT: sum += (*pArr)[i];
  // CHECK-NEXT: }
}
