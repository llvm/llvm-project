// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

// CHECK-LABEL: llvm.func @test_fake_use(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) {
// CHECK:         llvm.intr.fake.use %[[ARG0]] : !llvm.ptr
// CHECK:         llvm.return
// CHECK:       }

func.func @test_fake_use(%arg0: !fir.ref<i32>) {
  fir.fake_use %arg0 : !fir.ref<i32>
  return
}
