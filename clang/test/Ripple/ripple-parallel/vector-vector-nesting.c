// REQUIRES: target=hexagon{{.*}} || target-aarch64 || target-x86_64
// RUN: %clang -ffreestanding -S -Xclang -disable-llvm-passes -emit-llvm -fenable-ripple %s -o - | FileCheck %s
// RUN: %clang -ffreestanding -S -O2 -emit-llvm -fenable-ripple %s -Wall -Wextra -o - 2>&1 | FileCheck --check-prefix=NOWARN %s
// Check that the ripple pass handles the transformation
// RUN: %clang -ffreestanding -S -O2 -emit-llvm -fenable-ripple %s -Wall -Wextra -o - 2>&1 | FileCheck --check-prefix=NOWARN %s
// RUN: %clang -xc++ -ffreestanding -S -O2 -emit-llvm -fenable-ripple %s -Wall -Wextra -o - 2>&1 | FileCheck --check-prefix=NOWARN %s

#include "../ripple_test.h"

// NOWARN-NOT: warning:
// NOWARN-NOT: error:

// CHECK:         [[TMP6:%.*]] = call ptr @llvm.ripple.block.setshape.i{{32|64}}(i{{32|64}} 0, i{{32|64}} 32, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1, i{{32|64}} 1)
// CHECK-NEXT:    store ptr [[TMP6]], ptr [[BST:%.*]]
// CHECK:         [[TMP9:%.*]] = load ptr, ptr [[BST]]
// CHECK-NEXT:    [[TMP10:%.*]] = call i{{32|64}} @llvm.ripple.block.getsize.i{{32|64}}(ptr [[TMP9]], i{{32|64}} 1)
// CHECK-NEXT:    [[TMP11:%.*]] = load ptr, ptr [[BST]]
// CHECK-NEXT:    [[TMP12:%.*]] = call i{{32|64}} @llvm.ripple.block.getsize.i{{32|64}}(ptr [[TMP11]], i{{32|64}} 0)
// CHECK-NEXT:    [[MUL:%.*]] = mul i{{32|64}} [[TMP10]], [[TMP12]]
// CHECK-NEXT:    [[Trunk:%.*]] = trunc i{{32|64}} [[MUL]] to i16
// CHECK-NEXT:    store i16 [[Trunk]], ptr {{.*}}

void check(int32_t start, int64_t end, float *x,
           float *y, float *xpy) {
  ripple_block_t BST = ripple_set_block_shape(0, 32);
  #ifdef USING_PRAGMA
  #pragma ripple parallel Block(BST) Dims(0) NoRemainder
  #pragma ripple parallel Block(BST) Dims(1) NoRemainder
  #else
  ripple_parallel_full(BST, 0);
  ripple_parallel_full(BST, 1);
  #endif
  for (short i = start; i < end; ++i)
    xpy[i] = x[i] + y[i];
}
