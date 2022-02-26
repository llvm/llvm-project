// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

int s0(int a, int b) {
  int x = a + b;
  if (x > 0)
    x = 0;
  else
    x = 1;
  return x;
}

// CHECK: #[[loc2:loc[0-9]+]] = loc(fused["{{.*}}sourcelocation.cpp":4:8, "{{.*}}sourcelocation.cpp":4:12])
// CHECK: #[[loc3:loc[0-9]+]] = loc(fused["{{.*}}sourcelocation.cpp":4:15, "{{.*}}sourcelocation.cpp":4:19])
// CHECK: module  {
// CHECK:   func @s0(%arg0: i32 loc(fused["{{.*}}sourcelocation.cpp":4:8, "{{.*}}sourcelocation.cpp":4:12]), %arg1: i32 loc(fused["{{.*}}sourcelocation.cpp":4:15, "{{.*}}sourcelocation.cpp":4:19])) -> i32 {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["x", cinit] {alignment = 4 : i64} loc(#[[loc4:loc[0-9]+]])
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", paraminit] {alignment = 4 : i64} loc(#[[loc3]])
// CHECK:     %2 = cir.alloca i32, cir.ptr <i32>, ["a", paraminit] {alignment = 4 : i64} loc(#[[loc2]])
// CHECK:     cir.store %arg0, %2 : i32, cir.ptr <i32> loc(#[[loc5:loc[0-9]+]])
// CHECK:     cir.store %arg1, %1 : i32, cir.ptr <i32> loc(#[[loc5]])
// CHECK:     %3 = cir.load %2 : cir.ptr <i32>, i32 loc(#[[loc6:loc[0-9]+]])
// CHECK:     %4 = cir.load %1 : cir.ptr <i32>, i32 loc(#[[loc7:loc[0-9]+]])
// CHECK:     %5 = cir.binop(add, %3, %4) : i32 loc(#[[loc8:loc[0-9]+]])
// CHECK:     cir.store %5, %0 : i32, cir.ptr <i32> loc(#[[loc4]])
// CHECK:     cir.scope {
// CHECK:       %7 = cir.load %0 : cir.ptr <i32>, i32 loc(#[[loc10:loc[0-9]+]])
// CHECK:       %8 = cir.cst(0 : i32) : i32 loc(#[[loc11:loc[0-9]+]])
// CHECK:       %9 = cir.cmp(gt, %7, %8) : i32, !cir.bool loc(#[[loc12:loc[0-9]+]])
// CHECK:       cir.if %9 {
// CHECK:         %10 = cir.cst(0 : i32) : i32 loc(#[[loc14:loc[0-9]+]])
// CHECK:         cir.store %10, %0 : i32, cir.ptr <i32> loc(#[[loc15:loc[0-9]+]])
// CHECK:       } else {
// CHECK:         %10 = cir.cst(1 : i32) : i32 loc(#[[loc16:loc[0-9]+]])
// CHECK:         cir.store %10, %0 : i32, cir.ptr <i32> loc(#[[loc17:loc[0-9]+]])
// CHECK:       } loc(#[[loc13:loc[0-9]+]])
// CHECK:     } loc(#[[loc9:loc[0-9]+]])
// CHECK:     %6 = cir.load %0 : cir.ptr <i32>, i32 loc(#[[loc18:loc[0-9]+]])
// CHECK:     cir.return %6 : i32 loc(#[[loc19:loc[0-9]+]])
// CHECK:   } loc(#[[loc1:loc[0-9]+]])
// CHECK: } loc(#[[loc0:loc[0-9]+]])
// CHECK: #[[loc0]] = loc(unknown)
// CHECK: #[[loc1]] = loc(fused["{{.*}}sourcelocation.cpp":4:1, "{{.*}}sourcelocation.cpp":11:1])
// CHECK: #[[loc4]] = loc(fused["{{.*}}sourcelocation.cpp":5:3, "{{.*}}sourcelocation.cpp":5:15])
// CHECK: #[[loc5]] = loc("{{.*}}sourcelocation.cpp":4:22)
// CHECK: #[[loc6]] = loc("{{.*}}sourcelocation.cpp":5:11)
// CHECK: #[[loc7]] = loc("{{.*}}sourcelocation.cpp":5:15)
// CHECK: #[[loc8]] = loc(fused["{{.*}}sourcelocation.cpp":5:11, "{{.*}}sourcelocation.cpp":5:15])
// CHECK: #[[loc9]] = loc(fused["{{.*}}sourcelocation.cpp":6:3, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #[[loc10]] = loc("{{.*}}sourcelocation.cpp":6:7)
// CHECK: #[[loc11]] = loc("{{.*}}sourcelocation.cpp":6:11)
// CHECK: #[[loc12]] = loc(fused["{{.*}}sourcelocation.cpp":6:7, "{{.*}}sourcelocation.cpp":6:11])
// CHECK: #[[loc13]] = loc(fused["{{.*}}sourcelocation.cpp":7:5, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #[[loc14]] = loc("{{.*}}sourcelocation.cpp":7:9)
// CHECK: #[[loc15]] = loc(fused["{{.*}}sourcelocation.cpp":7:5, "{{.*}}sourcelocation.cpp":7:9])
// CHECK: #[[loc16]] = loc("{{.*}}sourcelocation.cpp":9:9)
// CHECK: #[[loc17]] = loc(fused["{{.*}}sourcelocation.cpp":9:5, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #[[loc18]] = loc("{{.*}}sourcelocation.cpp":10:10)
// CHECK: #[[loc19]] = loc(fused["{{.*}}sourcelocation.cpp":10:3, "{{.*}}sourcelocation.cpp":10:10])
