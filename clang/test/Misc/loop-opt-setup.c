// RUN: %clang -O1 -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s

extern int a[16];
int b = 0;
int foo(void) {
#pragma unroll
  for (int i = 0; i < 16; ++i)
    a[i] = b += 2;
  return b;
}
// Check br i1 to make sure that the loop is fully unrolled
// CHECK-LABEL: foo
// CHECK-NOT: br i1

void Helper() {
  const int *nodes[5];
  int num_active = 5;

  while (num_active)
#pragma clang loop unroll(full)
    for (int i = 0; i < 5; ++i)
      if (nodes[i])
        --num_active;
}

// Check br i1 to make sure the loop is gone, there will still be a label branch for the infinite loop.
// CHECK-LABEL: Helper
// CHECK:         br label
// CHECK-NOT:     br i1
// CHECK:         br label
