// Test the legacy mode of allocation-placement, which emulates the
// memory-allocation-opt pass, and the stack-arrays pass under -fstack-arrays,
// instead of using the size-based policy.

// RUN: fir-opt --allocation-placement="legacy=true" %s | FileCheck %s --check-prefixes=ALL,KEEP,OPT-OUT-KEEP
// RUN: fir-opt --allocation-placement="legacy=true dynamic-array-on-heap=true" %s | FileCheck %s --check-prefixes=ALL,DYN-HEAP,OPT-OUT-HEAP
// RUN: fir-opt --allocation-placement="legacy=true stack-arrays=true" %s | FileCheck %s --check-prefixes=ALL,STACK,OPT-OUT-KEEP
// RUN: fir-opt --allocation-placement="legacy=true stack-arrays=true dynamic-array-on-heap=true" %s | FileCheck %s --check-prefixes=ALL,STACK,OPT-OUT-HEAP

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {

func.func private @use_array(!fir.ref<!fir.array<?xi32>>)
func.func private @use_char(!fir.ref<!fir.char<1,?>>)

// The size-based policy would move this small temporary to the stack. Only
// -fstack-arrays does in legacy mode.
// ALL-LABEL: func.func @small_heap_temp
// KEEP: fir.allocmem !fir.array<10xi32>
// DYN-HEAP: fir.allocmem !fir.array<10xi32>
// STACK: fir.alloca !fir.array<10xi32>
// STACK-NOT: fir.allocmem
func.func @small_heap_temp() {
  %0 = fir.allocmem !fir.array<10xi32>
  %r = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi32>>
  fir.call @use_array(%r) : (!fir.ref<!fir.array<?xi32>>) -> ()
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  return
}

// The size-based policy would move this big temporary to the heap. Legacy mode
// has no size threshold.
// ALL-LABEL: func.func @big_stack_temp
// ALL: fir.alloca !fir.array<100000xi32>
// ALL-NOT: fir.allocmem
func.func @big_stack_temp() {
  %0 = fir.alloca !fir.array<100000xi32>
  %r = fir.convert %0 : (!fir.ref<!fir.array<100000xi32>>) -> !fir.ref<!fir.array<?xi32>>
  fir.call @use_array(%r) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

// Dynamic-size stack allocations only move to the heap under
// -fdynamic-heap-array, which -fstack-arrays overrides.
// ALL-LABEL: func.func @dynamic_stack_array
// KEEP: fir.alloca !fir.array<?xi32>
// KEEP-NOT: fir.allocmem
// DYN-HEAP: %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>
// DYN-HEAP: fir.call @use_array
// DYN-HEAP: fir.freemem %[[MEM]]
// STACK: fir.alloca !fir.array<?xi32>
// STACK-NOT: fir.allocmem
func.func @dynamic_stack_array(%n: index) {
  %0 = fir.alloca !fir.array<?xi32>, %n {bindc_name = "a", uniq_name = "_QFEa"}
  fir.call @use_array(%0) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

// -fdynamic-heap-array is not limited to arrays.
// ALL-LABEL: func.func @dynamic_stack_character
// KEEP: fir.alloca !fir.char<1,?>
// KEEP-NOT: fir.allocmem
// DYN-HEAP: %[[MEM:.*]] = fir.allocmem !fir.char<1,?>
// DYN-HEAP: fir.call @use_char
// DYN-HEAP: fir.freemem %[[MEM]]
// STACK: fir.alloca !fir.char<1,?>
// STACK-NOT: fir.allocmem
func.func @dynamic_stack_character(%n: index) {
  %0 = fir.alloca !fir.char<1,?>(%n : index) {bindc_name = "c", uniq_name = "_QFEc"}
  fir.call @use_char(%0) : (!fir.ref<!fir.char<1,?>>) -> ()
  return
}

// A function that opted out of -fstack-arrays in its own policy is handled as
// if -fstack-arrays was not set, so -fdynamic-heap-array applies to it.
// ALL-LABEL: func.func @opted_out_dynamic_stack_array
// OPT-OUT-KEEP: fir.alloca !fir.array<?xi32>
// OPT-OUT-KEEP-NOT: fir.allocmem
// OPT-OUT-HEAP: %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>
// OPT-OUT-HEAP: fir.call @use_array
// OPT-OUT-HEAP: fir.freemem %[[MEM]]
func.func @opted_out_dynamic_stack_array(%n: index) attributes {
    fir.allocation_policy = #fir.allocation_policy<stack_arrays = false,
        small_array_threshold = 1024, total_stack_limit = 4194304>} {
  %0 = fir.alloca !fir.array<?xi32>, %n {bindc_name = "a", uniq_name = "_QFEa"}
  fir.call @use_array(%0) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

// Allocations that must stay on the stack are never moved.
// ALL-LABEL: func.func @must_be_stack
// ALL: fir.alloca !fir.array<?xi32>
// ALL-NOT: fir.allocmem
func.func @must_be_stack(%n: index) {
  %0 = fir.alloca !fir.array<?xi32>, %n {bindc_name = "r", uniq_name = "_QFfEr", fir.must_be_stack = true}
  fir.call @use_array(%0) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}

// Like the stack-arrays pass, -fstack-arrays leaves heap allocations of
// offload regions alone.
// ALL-LABEL: func.func @dynamic_heap_temp_in_acc_parallel
// ALL: acc.parallel {
// ALL-NEXT: fir.allocmem !fir.array<?xi32>, %{{.*}}
// ALL: fir.freemem
// ALL-NOT: fir.alloca
func.func @dynamic_heap_temp_in_acc_parallel(%n: index) {
  acc.parallel {
    %0 = fir.allocmem !fir.array<?xi32>, %n
    %r = fir.convert %0 : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    fir.call @use_array(%r) : (!fir.ref<!fir.array<?xi32>>) -> ()
    fir.freemem %0 : !fir.heap<!fir.array<?xi32>>
    acc.yield
  }
  return
}

}
