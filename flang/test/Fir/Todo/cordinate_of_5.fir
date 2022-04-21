// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// CHECK: unsupported combination of coordinate operands
func.func @test_coordinate_of(%arr : !fir.ref<!fir.array<2 x !fir.char<10, 2>>>, %arg1: index) {
  %1 = arith.constant 10 : i32
  %2 = fir.coordinate_of %arr, %arg1, %1 : (!fir.ref<!fir.array<2 x !fir.char<10, 2>>>, index, i32) -> !fir.ref<!fir.char<1,10>>
  return
}
