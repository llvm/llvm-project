// Test that -fstack-arrays is not honored inside an offload region: the region
// runs on the device stack, which is far smaller than the host one, so its
// runtime-sized temporaries stay on the heap while the same allocation in host
// code goes on the stack. The size based part of the policy still applies, so
// a small constant-size temporary goes on the device stack.

// RUN: fir-opt --allocation-placement %s | FileCheck %s

module attributes {fir.allocation_policy =
                       #fir.allocation_policy<stack_arrays = true,
                           small_array_threshold = 1024,
                           total_stack_limit = 4194304>,
                   fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
                   llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {

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

// CHECK-LABEL: func.func @dynamic_temp_in_acc_kernels
// CHECK: acc.kernels {
// CHECK-NEXT: fir.allocmem !fir.array<?xi32>, %{{.*}}
// CHECK: fir.freemem
// CHECK-NOT: fir.alloca
func.func @dynamic_temp_in_acc_kernels(%n: index) {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  acc.kernels {
    %0 = fir.allocmem !fir.array<?xi32>, %n
    %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %v to %e : !fir.ref<i32>
    fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
    acc.terminator
  }
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

// CHECK-LABEL: func.func @dynamic_temp_in_acc_serial
// CHECK: acc.serial {
// CHECK-NEXT: fir.allocmem !fir.array<?xi32>, %{{.*}}
// CHECK: fir.freemem
// CHECK-NOT: fir.alloca
func.func @dynamic_temp_in_acc_serial(%n: index) {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  acc.serial {
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

// <10xi32> is 40 bytes, under the 1024 byte threshold: the size based policy
// still puts it on the (device) stack.
// CHECK-LABEL: func.func @small_temp_in_acc_parallel
// CHECK: acc.parallel {
// CHECK-NEXT: fir.alloca !fir.array<10xi32>
// CHECK-NOT: fir.allocmem
func.func @small_temp_in_acc_parallel() {
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  acc.parallel {
    %0 = fir.allocmem !fir.array<10xi32>
    %r = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
    %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
    fir.store %v to %e : !fir.ref<i32>
    fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
    acc.yield
  }
  return
}

}
