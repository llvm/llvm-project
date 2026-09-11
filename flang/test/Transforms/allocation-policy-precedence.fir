// Test where allocation-placement takes its policy from. The policy recorded
// on the module is the baseline; a pass option overrides it, but only when it
// is set explicitly, so that pinning one field in a test does not silently
// reset the others.
//
// i32 is 4 bytes, so <100xi32> is 400 bytes: on the stack under the default
// 1024 byte threshold, on the heap under a 64 byte one.

// RUN: fir-opt --allocation-placement -split-input-file %s | FileCheck %s
// RUN: fir-opt --allocation-placement="small-array-threshold=4096" \
// RUN:   -split-input-file %s | FileCheck %s --check-prefix=OPTION

// The policy recorded on the module is used when no option is given.

// CHECK-LABEL:   func.func @with_policy_attribute
// CHECK:           fir.allocmem !fir.array<100xi32>
// OPTION-LABEL:  func.func @with_policy_attribute
// OPTION:          fir.alloca !fir.array<100xi32>
// OPTION-NOT:      fir.allocmem
module attributes {fir.allocation_policy =
                       #fir.allocation_policy<stack_arrays = false,
                           small_array_threshold = 64,
                           total_stack_limit = 4194304>,
                   fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
                   llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {
func.func @with_policy_attribute() {
  %0 = fir.allocmem !fir.array<100xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<100xi32>>
  return
}
}

// -----

// Without a policy on the module the defaults apply, which put a 400 byte
// temporary on the stack. Both runs agree here, which is the point: the
// default in the option and the default in the policy are the same value.

// CHECK-LABEL:   func.func @without_policy_attribute
// CHECK:           fir.alloca !fir.array<100xi32>
// CHECK-NOT:       fir.allocmem
// OPTION-LABEL:  func.func @without_policy_attribute
// OPTION:          fir.alloca !fir.array<100xi32>
// OPTION-NOT:      fir.allocmem
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "",
                   llvm.data_layout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"} {
func.func @without_policy_attribute() {
  %0 = fir.allocmem !fir.array<100xi32>
  %c0 = arith.constant 0 : index
  %v = arith.constant 0 : i32
  %r = fir.convert %0 : (!fir.heap<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>>
  %e = fir.coordinate_of %r, %c0 : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
  fir.store %v to %e : !fir.ref<i32>
  fir.freemem %0 : !fir.heap<!fir.array<100xi32>>
  return
}
}
