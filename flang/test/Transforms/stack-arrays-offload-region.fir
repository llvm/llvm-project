// Test that stack-arrays leaves the heap allocations of an offload region and
// of an acc routine alone: they run on the device stack, which is far smaller
// than the host one. The same allocation in host code is moved to the stack.

// RUN: fir-opt --stack-arrays %s | FileCheck %s

// CHECK-LABEL: func.func @dynamic_temp_in_host_code
// CHECK: fir.alloca !fir.array<?xi32>, %{{.*}}
// CHECK-NOT: fir.allocmem
func.func @dynamic_temp_in_host_code(%n: index) {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %0 = fir.allocmem !fir.array<?xi32>, %n
  %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
  return
}

// CHECK-LABEL: func.func @dynamic_temp_in_acc_parallel
// CHECK: acc.parallel {
// CHECK-NEXT: fir.allocmem !fir.array<?xi32>, %{{.*}}
// CHECK: fir.freemem
// CHECK-NOT: fir.alloca
func.func @dynamic_temp_in_acc_parallel(%n: index) {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  acc.parallel {
    %0 = fir.allocmem !fir.array<?xi32>, %n
    %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %v to %e : !fir.ref<i32>
    fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
    acc.yield
  }
  return
}

// CHECK-LABEL: func.func @dynamic_temp_in_cuf_kernel
// CHECK: cuf.kernel
// CHECK-NEXT: fir.allocmem !fir.array<?xi32>, %{{.*}}
// CHECK: fir.freemem
// CHECK-NOT: fir.alloca
func.func @dynamic_temp_in_cuf_kernel(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %v = arith.constant 0 : i32
  cuf.kernel<<<%c1_i32, %c1_i32>>> (%i : index) = (%c1 : index) to (%n : index) step (%c1 : index) {
    %0 = fir.allocmem !fir.array<?xi32>, %n
    %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %v to %e : !fir.ref<i32>
    fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
    "fir.end"() : () -> ()
  }
  return
}

// An acc routine is compiled for the device as well: its allocations stay on
// the heap even before it is specialized.
// CHECK-LABEL: func.func @dynamic_temp_in_acc_routine
// CHECK: fir.allocmem !fir.array<?xi32>, %{{.*}}
// CHECK-NOT: fir.alloca
func.func @dynamic_temp_in_acc_routine(%n: index) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>} {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %0 = fir.allocmem !fir.array<?xi32>, %n
  %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
  return
}
