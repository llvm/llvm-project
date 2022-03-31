// RUN: tco --target=x86_64-unknown-linux-gnu --inline-all %s -o - | FileCheck %s

// CHECK-LABEL: @add
func @add(%a : i32, %b : i32) -> i32 {
  // CHECK: %[[add:.*]] = add i32
  %p = arith.addi %a, %b : i32
  // CHECK: ret i32 %[[add]]
  return %p : i32
}

// CHECK-LABEL: @test
func @test(%a : i32, %b : i32, %c : i32) -> i32 {
  // CHECK: %[[add:.*]] = add i32
  %m = fir.call @add(%a, %b) : (i32, i32) -> i32
  // CHECK: %[[mul:.*]] = mul i32 %[[add]],
  %n = arith.muli %m, %c : i32
  // CHECK: ret i32 %[[mul]]
  return %n : i32
}
