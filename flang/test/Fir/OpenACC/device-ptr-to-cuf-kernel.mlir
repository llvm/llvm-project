// RUN: fir-opt %s --acc-device-ptr-to-cuf-kernel -split-input-file | FileCheck %s

// A CUF kernel launched inside an acc.data region that maps a directly-addressed
// (static) array. The kernel argument is the mapped variable itself, so it is
// replaced by the acc.use_device result.
func.func @static_array() {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QFEa"}
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%sh) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  %2 = acc.copyin varPtr(%1 : !fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
  acc.data dataOperands(%2 : !fir.ref<!fir.array<100xi32>>) {
    cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%1) : (!fir.ref<!fir.array<100xi32>>)
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @static_array
// CHECK: %[[DECL:.*]] = fir.declare
// CHECK: acc.data
// CHECK: %[[DEV:.*]] = acc.use_device varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>)
// CHECK: acc.host_data dataOperands(%[[DEV]]
// CHECK: cuf.kernel_launch @kernel<<<{{.*}}>>>(%[[DEV]]) : (!fir.ref<!fir.array<100xi32>>)
// CHECK: acc.terminator

// -----

// A CUF kernel launched inside an acc.data region that maps a descriptor-based
// (allocatable) variable. OpenACC maps the descriptor, so the data address is
// recomputed as box_addr(load(<device descriptor>)) on the acc.use_device
// result.
func.func @descriptor_array() {
  %c1 = arith.constant 1 : i32
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "h", uniq_name = "_QFEh"}
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEh"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = acc.create varPtr(%1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "h"}
  acc.data dataOperands(%2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
    %3 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %4 = fir.box_addr %3 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    %5 = fir.convert %4 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%5) : (!fir.ref<!fir.array<?xi32>>)
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @descriptor_array
// CHECK: %[[DECL:.*]] = fir.declare
// CHECK: %[[DEV:.*]] = acc.use_device varPtr(%[[DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK: acc.host_data dataOperands(%[[DEV]]
// CHECK: %[[LOAD:.*]] = fir.load %[[DEV]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK: %[[CONV:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: cuf.kernel_launch @kernel<<<{{.*}}>>>(%[[CONV]]) : (!fir.ref<!fir.array<?xi32>>)

// -----

// No enclosing acc.data region: the launch must be left untouched.
func.func @no_enclosing_data() {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca !fir.array<100xi32> {uniq_name = "_QFEa"}
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%sh) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%1) : (!fir.ref<!fir.array<100xi32>>)
  return
}

// CHECK-LABEL: func.func @no_enclosing_data
// CHECK-NOT: acc.use_device
// CHECK-NOT: acc.host_data
// CHECK: cuf.kernel_launch @kernel

// -----

// The enclosing acc.data maps a different variable than the one launched: no
// rewrite.
func.func @unmapped_arg() {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : index
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %0 = fir.alloca !fir.array<100xi32> {uniq_name = "_QFEa"}
  %1 = fir.declare %0(%sh) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  %2 = fir.alloca !fir.array<100xi32> {uniq_name = "_QFEb"}
  %3 = fir.declare %2(%sh) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  %4 = acc.copyin varPtr(%3 : !fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
  acc.data dataOperands(%4 : !fir.ref<!fir.array<100xi32>>) {
    cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%1) : (!fir.ref<!fir.array<100xi32>>)
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @unmapped_arg
// CHECK-NOT: acc.use_device
// CHECK-NOT: acc.host_data
