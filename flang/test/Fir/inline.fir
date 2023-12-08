// RUN: tco --inline-all %s -o - | FileCheck %s
// RUN: %flang_fc1  -mmlir --inline-all -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @add
func.func @add(%a : i32, %b : i32) -> i32 {
  // CHECK: %[[add:.*]] = add i32
  %p = arith.addi %a, %b : i32
  // CHECK: ret i32 %[[add]]
  return %p : i32
}

// CHECK-LABEL: @test
func.func @test(%a : i32, %b : i32, %c : i32) -> i32 {
  // CHECK: %[[add:.*]] = add i32
  %m = fir.call @add(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[mul:.*]] = mul i32 %[[add]],
  %n = arith.muli %m, %c : i32
  // CHECK: ret i32 %[[mul]]
  return %n : i32
}
