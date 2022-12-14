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

// CHECK: #loc2 = loc(fused["{{.*}}sourcelocation.cpp":4:8, "{{.*}}sourcelocation.cpp":4:12])
// CHECK: #loc3 = loc(fused["{{.*}}sourcelocation.cpp":4:15, "{{.*}}sourcelocation.cpp":4:19])
// CHECK: module {{.*}} {
// CHECK:   cir.func @_Z2s0ii(%arg0: i32 loc(fused["{{.*}}sourcelocation.cpp":4:8, "{{.*}}sourcelocation.cpp":4:12]), %arg1: i32 loc(fused["{{.*}}sourcelocation.cpp":4:15, "{{.*}}sourcelocation.cpp":4:19])) -> i32 {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64} loc(#loc2)
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", init] {alignment = 4 : i64} loc(#loc3)
// CHECK:     %2 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64} loc(#loc4)
// CHECK:     %3 = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64} loc(#loc5)
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32> loc(#loc6)
// CHECK:     cir.store %arg1, %1 : i32, cir.ptr <i32> loc(#loc6)
// CHECK:     %4 = cir.load %0 : cir.ptr <i32>, i32 loc(#loc7)
// CHECK:     %5 = cir.load %1 : cir.ptr <i32>, i32 loc(#loc8)
// CHECK:     %6 = cir.binop(add, %4, %5) : i32 loc(#loc9)
// CHECK:     cir.store %6, %3 : i32, cir.ptr <i32> loc(#loc5)
// CHECK:     cir.scope {
// CHECK:       %9 = cir.load %3 : cir.ptr <i32>, i32 loc(#loc11)
// CHECK:       %10 = cir.cst(0 : i32) : i32 loc(#loc12)
// CHECK:       %11 = cir.cmp(gt, %9, %10) : i32, !cir.bool loc(#loc13)
// CHECK:       cir.if %11 {
// CHECK:         %12 = cir.cst(0 : i32) : i32 loc(#loc15)
// CHECK:         cir.store %12, %3 : i32, cir.ptr <i32> loc(#loc16)
// CHECK:       } else {
// CHECK:         %12 = cir.cst(1 : i32) : i32 loc(#loc17)
// CHECK:         cir.store %12, %3 : i32, cir.ptr <i32> loc(#loc18)
// CHECK:       } loc(#loc14)
// CHECK:     } loc(#loc10)
// CHECK:     %7 = cir.load %3 : cir.ptr <i32>, i32 loc(#loc19)
// CHECK:     cir.store %7, %2 : i32, cir.ptr <i32> loc(#loc20)
// CHECK:     %8 = cir.load %2 : cir.ptr <i32>, i32 loc(#loc20)
// CHECK:     cir.return %8 : i32 loc(#loc20)
// CHECK:   } loc(#loc1)
// CHECK: } loc(#loc0)
// CHECK: #loc0 = loc(unknown)
// CHECK: #loc1 = loc(fused["{{.*}}sourcelocation.cpp":4:1, "{{.*}}sourcelocation.cpp":11:1])
// CHECK: #loc4 = loc("{{.*}}sourcelocation.cpp":11:1)
// CHECK: #loc5 = loc(fused["{{.*}}sourcelocation.cpp":5:3, "{{.*}}sourcelocation.cpp":5:15])
// CHECK: #loc6 = loc("{{.*}}sourcelocation.cpp":4:22)
// CHECK: #loc7 = loc("{{.*}}sourcelocation.cpp":5:11)
// CHECK: #loc8 = loc("{{.*}}sourcelocation.cpp":5:15)
// CHECK: #loc9 = loc(fused["{{.*}}sourcelocation.cpp":5:11, "{{.*}}sourcelocation.cpp":5:15])
// CHECK: #loc10 = loc(fused["{{.*}}sourcelocation.cpp":6:3, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #loc11 = loc("{{.*}}sourcelocation.cpp":6:7)
// CHECK: #loc12 = loc("{{.*}}sourcelocation.cpp":6:11)
// CHECK: #loc13 = loc(fused["{{.*}}sourcelocation.cpp":6:7, "{{.*}}sourcelocation.cpp":6:11])
// CHECK: #loc14 = loc(fused["{{.*}}sourcelocation.cpp":7:5, "{{.*}}sourcelocation.cpp":7:9, "{{.*}}sourcelocation.cpp":9:5, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #loc15 = loc("{{.*}}sourcelocation.cpp":7:9)
// CHECK: #loc16 = loc(fused["{{.*}}sourcelocation.cpp":7:5, "{{.*}}sourcelocation.cpp":7:9])
// CHECK: #loc17 = loc("{{.*}}sourcelocation.cpp":9:9)
// CHECK: #loc18 = loc(fused["{{.*}}sourcelocation.cpp":9:5, "{{.*}}sourcelocation.cpp":9:9])
// CHECK: #loc19 = loc("{{.*}}sourcelocation.cpp":10:10)
// CHECK: #loc20 = loc(fused["{{.*}}sourcelocation.cpp":10:3, "{{.*}}sourcelocation.cpp":10:10])
