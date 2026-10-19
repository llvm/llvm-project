// RUN: fir-opt %s --cse --split-input-file | FileCheck %s --check-prefix=CSE
// RUN: fir-opt %s --fir-memref-dataflow-opt --split-input-file | FileCheck %s --check-prefix=FORWARD

// Checks touch no user memory, so they do not prevent load CSE.
// CSE-LABEL: func.func @load_cse(
// CSE: %[[V:.*]] = fir.load
// CSE-NEXT: fir.assert
// CSE-NOT: fir.load
// CSE: arith.addi %[[V]], %[[V]]
func.func @load_cse(%addr: !fir.ref<i32>, %ok: i1) -> i32 {
  %a = fir.load %addr : !fir.ref<i32>
  fir.assert %ok, argument, "invalid argument"
  %b = fir.load %addr : !fir.ref<i32>
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}

// -----

// FORWARD-LABEL: func.func @store_forward(
// FORWARD-SAME: %[[ADDR:.*]]: !fir.ref<i32>, %[[VALUE:.*]]: i32,
// FORWARD: fir.store %[[VALUE]]
// FORWARD-NEXT: fir.assert
// FORWARD-NOT: fir.load
// FORWARD: return %[[VALUE]]
func.func @store_forward(%addr: !fir.ref<i32>, %value: i32, %ok: i1) -> i32 {
  fir.store %value to %addr : !fir.ref<i32>
  fir.assert %ok, argument, "invalid argument"
  %load = fir.load %addr : !fir.ref<i32>
  return %load : i32
}

// -----

// CSE-LABEL: func.func @guarded_load(
// CSE: fir.load
// CSE: %[[G:.*]] = fir.assert
// CSE-NEXT: %{{.*}} = fir.load %[[G]]
func.func @guarded_load(%addr: !fir.ref<i32>, %ok: i1) -> i32 {
  %a = fir.load %addr : !fir.ref<i32>
  %guarded = fir.assert %ok, pointer, "invalid pointer" guard(%addr : !fir.ref<i32>)
  %b = fir.load %guarded : !fir.ref<i32>
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}
