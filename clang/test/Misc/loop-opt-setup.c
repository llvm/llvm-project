// This tests loop unrolling and loop deletion (enabled under -O1)
// RUN: %clang_cc1 -std=c11 -O1 -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s
// RUN: %clang_cc1 -std=c99 -O1 -fno-unroll-loops -S -o - %s -emit-llvm | FileCheck %s --check-prefix C99

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
// In C99, there was no forward progress requirement, so we expect the infinite loop to still exist,
// but for C11 and onwards, the infinite loop can be deleted.
// CHECK-LABEL: Helper
// C99: br label
// C99-NOT: br i1
// C99: br label
// CHECK: entry:
// CHECK-NOT: br i1
// CHECK-NEXT: ret void
