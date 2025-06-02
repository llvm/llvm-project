// REQUIRES: asserts
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -debug-only=fir-alias-analysis 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "_QFPtest"

// Checking that the source kind of a load of a load is SourceKind::Indirect
// CHECK: {test.ptr = "load_load"}
// CHECK-NEXT: SourceKind: Indirect

// Checking that the source kind of a load of an arg is SourceKind::Argument
// CHECK: {test.ptr = "load_arg"}
// CHECK-NEXT: SourceKind: Argument

func.func @_QFPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> ) attributes {test.ptr = "func"} {
  
  %0 = fir.alloca !fir.llvm_ptr<!fir.box<!fir.ptr<f32>>>
  %1 = fir.convert %arg0 : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.llvm_ptr<!fir.box<!fir.ptr<f32>>>
  fir.store %1 to %0 : !fir.ref<!fir.llvm_ptr<!fir.box<!fir.ptr<f32>>>>
  %2 = fir.load %0 : !fir.ref<!fir.llvm_ptr<!fir.box<!fir.ptr<f32>>>>
  %3 = fir.convert %2 : (!fir.llvm_ptr<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %15 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %16 = fir.box_addr %15 {test.ptr = "load_load"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %17 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %18 = fir.box_addr %17 {test.ptr = "load_arg"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}