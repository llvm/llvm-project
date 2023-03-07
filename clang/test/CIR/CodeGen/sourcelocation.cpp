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

// CHECK: #loc3 = loc("{{.*}}sourcelocation.cpp":4:8)
// CHECK: #loc4 = loc("{{.*}}sourcelocation.cpp":4:12)
// CHECK: #loc5 = loc("{{.*}}sourcelocation.cpp":4:15)
// CHECK: #loc6 = loc("{{.*}}sourcelocation.cpp":4:19)
// CHECK: #loc21 = loc(fused[#loc3, #loc4])
// CHECK: #loc22 = loc(fused[#loc5, #loc6])
// CHECK: module attributes {cir.sob = #cir.signed_overflow_behavior<undefined>} {
// CHECK:   cir.func @_Z2s0ii(%arg0: i32 loc(fused[#loc3, #loc4]), %arg1: i32 loc(fused[#loc5, #loc6])) -> i32 {
// CHECK:     %0 = cir.alloca i32, cir.ptr <i32>, ["a", init] {alignment = 4 : i64} loc(#loc21)
// CHECK:     %1 = cir.alloca i32, cir.ptr <i32>, ["b", init] {alignment = 4 : i64} loc(#loc22)
// CHECK:     %2 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64} loc(#loc2)
// CHECK:     %3 = cir.alloca i32, cir.ptr <i32>, ["x", init] {alignment = 4 : i64} loc(#loc23)
// CHECK:     cir.store %arg0, %0 : i32, cir.ptr <i32> loc(#loc9)
// CHECK:     cir.store %arg1, %1 : i32, cir.ptr <i32> loc(#loc9)
// CHECK:     %4 = cir.load %0 : cir.ptr <i32>, i32 loc(#loc10)
// CHECK:     %5 = cir.load %1 : cir.ptr <i32>, i32 loc(#loc8)
// CHECK:     %6 = cir.binop(add, %4, %5) : i32 loc(#loc24)
// CHECK:     cir.store %6, %3 : i32, cir.ptr <i32> loc(#loc23)
// CHECK:     cir.scope {
// CHECK:       %9 = cir.load %3 : cir.ptr <i32>, i32 loc(#loc13)
// CHECK:       %10 = cir.const(0 : i32) : i32 loc(#loc14)
// CHECK:       %11 = cir.cmp(gt, %9, %10) : i32, !cir.bool loc(#loc26)
// CHECK:       cir.if %11 {
// CHECK:         %12 = cir.const(0 : i32) : i32 loc(#loc16)
// CHECK:         cir.store %12, %3 : i32, cir.ptr <i32> loc(#loc28)
// CHECK:       } else {
// CHECK:         %12 = cir.const(1 : i32) : i32 loc(#loc12)
// CHECK:         cir.store %12, %3 : i32, cir.ptr <i32> loc(#loc29)
// CHECK:       } loc(#loc27)
// CHECK:     } loc(#loc25)
// CHECK:     %7 = cir.load %3 : cir.ptr <i32>, i32 loc(#loc18)
// CHECK:     cir.store %7, %2 : i32, cir.ptr <i32> loc(#loc30)
// CHECK:     %8 = cir.load %2 : cir.ptr <i32>, i32 loc(#loc30)
// CHECK:     cir.return %8 : i32 loc(#loc30)
// CHECK:   } loc(#loc20)
// CHECK: } loc(#loc)
// CHECK: #loc = loc(unknown)
// CHECK: #loc1 = loc("{{.*}}sourcelocation.cpp":4:1)
// CHECK: #loc2 = loc("{{.*}}sourcelocation.cpp":11:1)
// CHECK: #loc7 = loc("{{.*}}sourcelocation.cpp":5:3)
// CHECK: #loc8 = loc("{{.*}}sourcelocation.cpp":5:15)
// CHECK: #loc9 = loc("{{.*}}sourcelocation.cpp":4:22)
// CHECK: #loc10 = loc("{{.*}}sourcelocation.cpp":5:11)
// CHECK: #loc11 = loc("{{.*}}sourcelocation.cpp":6:3)
// CHECK: #loc12 = loc("{{.*}}sourcelocation.cpp":9:9)
// CHECK: #loc13 = loc("{{.*}}sourcelocation.cpp":6:7)
// CHECK: #loc14 = loc("{{.*}}sourcelocation.cpp":6:11)
// CHECK: #loc15 = loc("{{.*}}sourcelocation.cpp":7:5)
// CHECK: #loc16 = loc("{{.*}}sourcelocation.cpp":7:9)
// CHECK: #loc17 = loc("{{.*}}sourcelocation.cpp":9:5)
// CHECK: #loc18 = loc("{{.*}}sourcelocation.cpp":10:10)
// CHECK: #loc19 = loc("{{.*}}sourcelocation.cpp":10:3)
// CHECK: #loc20 = loc(fused[#loc1, #loc2])
// CHECK: #loc23 = loc(fused[#loc7, #loc8])
// CHECK: #loc24 = loc(fused[#loc10, #loc8])
// CHECK: #loc25 = loc(fused[#loc11, #loc12])
// CHECK: #loc26 = loc(fused[#loc13, #loc14])
// CHECK: #loc27 = loc(fused[#loc15, #loc16, #loc17, #loc12])
// CHECK: #loc28 = loc(fused[#loc15, #loc16])
// CHECK: #loc29 = loc(fused[#loc17, #loc12])
// CHECK: #loc30 = loc(fused[#loc19, #loc18])
