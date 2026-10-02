// RUN: fir-opt %s --acc-device-ptr-to-cuf-kernel -split-input-file | FileCheck %s

// A CUF kernel launched inside an acc.data region that maps a directly-addressed
// (static) array. The kernel argument is the mapped variable itself, so it is
// wrapped in acc.use_device and substituted directly.
func.func @static_array() {
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QFEa"}
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%sh) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  %2 = acc.copyin varPtr(%1 : !fir.ref<!fir.array<100xi32>>) name("a") -> !fir.ref<!fir.array<100xi32>>
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
// CHECK: ifPresent

// -----

// A CUF kernel launched with an array section. The launch argument is already an
// interior pointer (&a(3)) computed on the host base. The pass wraps that
// interior pointer directly in acc.use_device and substitutes it; the section
// addressing is left untouched (not rebuilt on a device base).
func.func @array_section() {
  %c1 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c100 = arith.constant 100 : index
  %c3 = arith.constant 3 : index
  %0 = fir.alloca !fir.array<100xi32> {uniq_name = "_QFEa"}
  %sh = fir.shape %c100 : (index) -> !fir.shape<1>
  %1 = fir.declare %0(%sh) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
  %2 = acc.copyin varPtr(%1 : !fir.ref<!fir.array<100xi32>>) name("a") -> !fir.ref<!fir.array<100xi32>>
  acc.data dataOperands(%2 : !fir.ref<!fir.array<100xi32>>) {
    %3 = fir.array_coor %1(%sh) %c3 : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %4 = fir.convert %3 : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
    cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%4, %c2_i32) : (!fir.ref<!fir.array<?xi32>>, i32)
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @array_section
// CHECK: %[[DECL:.*]] = fir.declare
// CHECK: acc.data
// CHECK: %[[COOR:.*]] = fir.array_coor %[[DECL]]
// CHECK: %[[CONV:.*]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: %[[DEV:.*]] = acc.use_device varPtr(%[[CONV]] : !fir.ref<!fir.array<?xi32>>)
// CHECK: acc.host_data dataOperands(%[[DEV]]
// CHECK: cuf.kernel_launch @kernel<<<{{.*}}>>>(%[[DEV]], {{.*}}) : (!fir.ref<!fir.array<?xi32>>, i32)
// CHECK: ifPresent

// -----

// A CUF kernel launched inside an acc.data region that maps a descriptor-based
// (allocatable) variable. OpenACC maps the descriptor, but the launch already
// extracts the data address on the host via box_addr(load(<descriptor>)). The
// pass wraps that data pointer in acc.use_device and substitutes it directly;
// the descriptor addressing is left on the host descriptor and not rebuilt.
func.func @descriptor_array() {
  %c1 = arith.constant 1 : i32
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "h", uniq_name = "_QFEh"}
  %1 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEh"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = acc.create varPtr(%1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) name("h") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
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
// CHECK: acc.data
// CHECK: %[[LOAD:.*]] = fir.load %[[DECL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK: %[[CONV:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: %[[DEV:.*]] = acc.use_device varPtr(%[[CONV]] : !fir.ref<!fir.array<?xi32>>)
// CHECK: acc.host_data dataOperands(%[[DEV]]
// CHECK: cuf.kernel_launch @kernel<<<{{.*}}>>>(%[[DEV]]) : (!fir.ref<!fir.array<?xi32>>)
// CHECK: ifPresent

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
  %4 = acc.copyin varPtr(%3 : !fir.ref<!fir.array<100xi32>>) name("b") -> !fir.ref<!fir.array<100xi32>>
  acc.data dataOperands(%4 : !fir.ref<!fir.array<100xi32>>) {
    cuf.kernel_launch @kernel<<<%c1, %c1, %c1, %c1, %c1, %c1>>>(%1) : (!fir.ref<!fir.array<100xi32>>)
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @unmapped_arg
// CHECK-NOT: acc.use_device
// CHECK-NOT: acc.host_data
