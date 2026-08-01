// RUN: fir-opt %s --strict-fir-volatile-verifier --fir-to-memref -cg-rewrite --fir-to-llvm-ir | FileCheck %s

// Volatile accesses skipped by fir-to-memref still reach LLVM IR through the
// FIR code generation path, which sets the volatile flag from the reference
// type.

// CHECK-LABEL: llvm.func @volatile_roundtrip
// CHECK:         %[[LOAD:.*]] = llvm.load volatile %{{.*}} : !llvm.ptr -> f32
// CHECK:         llvm.store volatile %[[LOAD]], %{{.*}} : f32, !llvm.ptr
func.func @volatile_roundtrip(%arg0: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.volatile_cast %arg0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "x"} : (!fir.ref<f32, volatile>, !fir.dscope) -> !fir.ref<f32, volatile>
  %3 = fir.load %2 : !fir.ref<f32, volatile>
  fir.store %3 to %2 : !fir.ref<f32, volatile>
  return
}
