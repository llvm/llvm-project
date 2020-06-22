// RUN: %clang -O1 -fexperimental-new-pass-manager -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s -check-prefix=CHECK-NEWPM
// RUN: %clang -O1 -fno-experimental-new-pass-manager -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s -check-prefix=CHECK-OLDPM
extern int a[16];
int b = 0;
int foo(void) {
#pragma unroll
  for (int i = 0; i < 16; ++i)
    a[i] = b += 2;
  return b;
}
// Check br i1 to make sure that the loop is fully unrolled
// CHECK-LABEL-NEWPM: foo
// CHECK-NOT-NEWPM: br i1
// CHECK-LABEL-OLDPM: foo
// CHECK-NOT-OLDPM: br i1

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
// CHECK-LABEL-NEWPM: Helper
// CHECK-NEWPM: br label
// CHECK-NEWPM-NOT: br i1
// CHECK-NEWPM: br label

// The old pass manager doesn't remove the while loop so check for 5 load i32*.
// CHECK-LABEL-OLDPM: Helper
// CHECK-OLDPM: br label
// CHECK-OLDPM: load i32*
// CHECK-OLDPM: load i32*
// CHECK-OLDPM: load i32*
// CHECK-OLDPM: load i32*
// CHECK-OLDPM: load i32*
// CHECK-OLDPM: ret
