// RUN: fir-opt %s --acc-hoist-allocatable-realloc -split-input-file | FileCheck %s

// The separated reallocation is hoisted out of acc kernels; only the
// assignment stays on the device.

// CHECK-LABEL: func.func @hoist
// CHECK: fir.allocmem {{.*}} {uniq_name = ".auto.alloc"}
// CHECK: fir.store {{.*}} to %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: acc.kernels {
// CHECK-NOT: fir.allocmem
// CHECK: hlfir.assign
// CHECK: acc.terminator
func.func @hoist(%b: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %n: index, %cond: i1, %v: f32) {
  acc.kernels {
    %m = fir.if %cond -> !fir.heap<!fir.array<?xf32>> {
      %a = fir.allocmem !fir.array<?xf32>, %n {uniq_name = ".auto.alloc"}
      fir.result %a : !fir.heap<!fir.array<?xf32>>
    } else {
      %z = fir.zero_bits !fir.heap<!fir.array<?xf32>>
      fir.result %z : !fir.heap<!fir.array<?xf32>>
    }
    fir.if %cond {
      %sh = fir.shape %n : (index) -> !fir.shape<1>
      %box = fir.embox %m(%sh) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
      fir.store %box to %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    }
    %ld = fir.load %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    hlfir.assign %v to %ld : f32, !fir.box<!fir.heap<!fir.array<?xf32>>>
    acc.terminator
  }
  return
}

// -----

// The RHS reads the old storage that the reallocation frees, so it must stay
// inside the kernel (e.g. `b = b + 1`).

// CHECK-LABEL: func.func @selfdep_no_hoist
// CHECK: acc.kernels {
// CHECK: fir.allocmem {{.*}} {uniq_name = ".auto.alloc"}
// CHECK: acc.terminator
func.func @selfdep_no_hoist(%b: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %n: index, %cond: i1, %v: f32) {
  acc.kernels {
    %old = fir.load %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %m = fir.if %cond -> !fir.heap<!fir.array<?xf32>> {
      %a = fir.allocmem !fir.array<?xf32>, %n {uniq_name = ".auto.alloc"}
      fir.result %a : !fir.heap<!fir.array<?xf32>>
    } else {
      %z = fir.zero_bits !fir.heap<!fir.array<?xf32>>
      fir.result %z : !fir.heap<!fir.array<?xf32>>
    }
    fir.if %cond {
      %sh = fir.shape %n : (index) -> !fir.shape<1>
      %box = fir.embox %m(%sh) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
      fir.store %box to %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    }
    %new = fir.load %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    hlfir.assign %old to %new : !fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>
    acc.terminator
  }
  return
}

// -----

// acc parallel reallocation is the user's responsibility; only acc kernels is
// hoisted.

// CHECK-LABEL: func.func @parallel_no_hoist
// CHECK: acc.parallel {
// CHECK: fir.allocmem {{.*}} {uniq_name = ".auto.alloc"}
// CHECK: acc.yield
func.func @parallel_no_hoist(%b: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %n: index, %cond: i1, %v: f32) {
  acc.parallel {
    %m = fir.if %cond -> !fir.heap<!fir.array<?xf32>> {
      %a = fir.allocmem !fir.array<?xf32>, %n {uniq_name = ".auto.alloc"}
      fir.result %a : !fir.heap<!fir.array<?xf32>>
    } else {
      %z = fir.zero_bits !fir.heap<!fir.array<?xf32>>
      fir.result %z : !fir.heap<!fir.array<?xf32>>
    }
    fir.if %cond {
      %sh = fir.shape %n : (index) -> !fir.shape<1>
      %box = fir.embox %m(%sh) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
      fir.store %box to %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    }
    %ld = fir.load %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    hlfir.assign %v to %ld : f32, !fir.box<!fir.heap<!fir.array<?xf32>>>
    acc.yield
  }
  return
}

// -----

// An in-loop reallocation (nested in a loop, not at the construct's outer
// level) must stay inside the kernel.

// CHECK-LABEL: func.func @inloop_no_hoist
// CHECK: acc.kernels {
// CHECK: fir.do_loop
// CHECK: fir.allocmem {{.*}} {uniq_name = ".auto.alloc"}
// CHECK: acc.terminator
func.func @inloop_no_hoist(%b: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %n: index, %cond: i1, %v: f32, %lb: index, %ub: index, %step: index) {
  acc.kernels {
    fir.do_loop %i = %lb to %ub step %step {
      %m = fir.if %cond -> !fir.heap<!fir.array<?xf32>> {
        %a = fir.allocmem !fir.array<?xf32>, %n {uniq_name = ".auto.alloc"}
        fir.result %a : !fir.heap<!fir.array<?xf32>>
      } else {
        %z = fir.zero_bits !fir.heap<!fir.array<?xf32>>
        fir.result %z : !fir.heap<!fir.array<?xf32>>
      }
      fir.if %cond {
        %sh = fir.shape %n : (index) -> !fir.shape<1>
        %box = fir.embox %m(%sh) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
        fir.store %box to %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      }
      %ld = fir.load %b : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
      hlfir.assign %v to %ld : f32, !fir.box<!fir.heap<!fir.array<?xf32>>>
    }
    acc.terminator
  }
  return
}
