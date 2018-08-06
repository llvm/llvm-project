// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// We expect to get a loop structure like this:
//    do.body:                                       ; preds = %do.cond, ...
//      ...
//      br label %do.cond
//    do.cond:                                       ; preds = %do.body
//      ...
//      br i1 %cmp, label %do.body, label %do.end
//    do.end:                                        ; preds = %do.cond
//      ...
//
// Verify that the loop metadata only is put on the backedge.
//
// CHECK-NOT: llvm.loop
// CHECK-LABEL: do.cond:
// CHECK: br {{.*}}, label %do.body, label %do.end, !llvm.loop ![[LMD1:[0-9]+]]
// CHECK-LABEL: do.end:
// CHECK-NOT: llvm.loop
// CHECK: ![[LMD1]] = distinct !{![[LMD1]], ![[LMD2:[0-9]+]]}
// CHECK: ![[LMD2]] = !{!"llvm.loop.unroll.count", i32 4}

int test(int a[], int n) {
  int i = 0;
  int sum = 0;

#pragma unroll 4
  do
  {
    a[i] = a[i] + 1;
    sum = sum + a[i];
    i++;
  } while (i < n);

  return sum;
}
