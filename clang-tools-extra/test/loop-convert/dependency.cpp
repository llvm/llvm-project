// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: loop-convert . %t.cpp -- && FileCheck -input-file=%t.cpp %s

void f() {
  const int N = 6;
  const int M = 8;
  int arr[N][M];

  for (int i = 0; i < N; ++i) {
    int a = 0;
    int b = arr[i][a];
  }
  // CHECK: for (auto & [[VAR:[a-z_]+]] : arr) {
  // CHECK-NEXT: int a = 0;
  // CHECK-NEXT: int b = [[VAR]][a];
  // CHECK-NEXT: }

  for (int j = 0; j < M; ++j) {
    int a = 0;
    int b = arr[a][j];
  }
  // CHECK: for (int j = 0; j < M; ++j) {
  // CHECK-NEXT: int a = 0;
  // CHECK-NEXT: int b = arr[a][j];
  // CHECK-NEXT: }
}
