// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// CHECK: unsupported combination of coordinate operands

func.func @test_coordinate_of(%arr : !fir.ref<!fir.array<2 x i32>>, %arg1: index) {
  %2 = fir.coordinate_of %arr, %arg1, %arg1 : (!fir.ref<!fir.array<2 x i32>>, index, index) -> !fir.ref<i32>
  return
}
