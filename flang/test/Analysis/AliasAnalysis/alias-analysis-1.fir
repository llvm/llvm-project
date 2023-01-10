// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "_QPtest"
// CHECK-DAG: alloca_1#0 <-> address_of#0: NoAlias
func.func @_QPtest(%arg1: !fir.ref<i32>) {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca () -> () {test.ptr = "alloca_1"}
  %1 = fir.address_of(@_QPf) {test.ptr = "address_of"} : () -> i32 
  %2 = fir.convert %1 : (() -> i32) -> (() -> ())
  %4 = fir.convert %0 : (!fir.ref<() -> ()>) -> !fir.llvm_ptr<() -> ()>
  fir.store %2 to %4 : !fir.llvm_ptr<() -> ()>
  %6 = fir.load %0 : !fir.ref<() -> ()>
  fir.call @_QPs(%6) : (() -> ()) -> ()
  return
}

func.func private @_QPs(%arg0: () -> ()) 
func.func private @_QPf() -> i32
