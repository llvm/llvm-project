// RUN: fir-opt -cuf-alloc-delay --split-input-file %s | FileCheck %s

// Test 1: Basic delay — cuf.alloc of box type is moved to just before
// the first use (AllocatableSetBounds).
func.func @_QPbasic() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFbasicEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFbasicEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %c10 : (index) -> i64
  fir.call @_FortranAAllocatableSetBounds(%4, %5, %6, %6) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i32, i64, i64) -> ()
  %7 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  return
}
func.func private @_FortranAAllocatableSetBounds(!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i32, i64, i64)

// CHECK-LABEL: func.func @_QPbasic()
// The embox/zero_bits/shape stay at entry; cuf.alloc+store+declare move to
// just before the first use (AllocatableSetBounds).
// CHECK:         fir.zero_bits
// CHECK:         fir.shape
// CHECK:         fir.embox
// CHECK:         arith.constant 10
// CHECK:         arith.constant 1
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         fir.call @_FortranAAllocatableSetBounds
// CHECK:         cuf.allocate

// -----

// Test 2: Non-box cuf.alloc (fixed-size device array) should NOT be moved.
func.func @_QPfixed_array() {
  %0 = cuf.alloc !fir.array<10xf32> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFfixed_arrayEa"} -> !fir.ref<!fir.array<10xf32>>
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.declare %0(%1) {data_attr = #cuf.cuda<device>, uniq_name = "_QFfixed_arrayEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
  %c42 = arith.constant 42 : index
  return
}

// CHECK-LABEL: func.func @_QPfixed_array()
// CHECK:         cuf.alloc !fir.array<10xf32>
// CHECK:         arith.constant 10
// CHECK:         fir.shape
// CHECK:         fir.declare
// CHECK:         arith.constant 42

// -----

// Test 3: First use inside fir.if — cuf.alloc should be moved before the
// fir.if, not inside it.
func.func @_QPuse_in_if(%arg0: !fir.ref<!fir.logical<4>>) {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFuse_in_ifEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFuse_in_ifEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c42 = arith.constant 42 : index
  %5 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %6 = fir.convert %5 : (!fir.logical<4>) -> i1
  fir.if %6 {
    %7 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  }
  return
}

// CHECK-LABEL: func.func @_QPuse_in_if
// cuf.alloc+store+declare move to just before fir.if (not inside it).
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 42
// CHECK:         fir.load
// CHECK:         fir.convert
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         fir.if
// CHECK:           cuf.allocate

// -----

// Test 4: cuf.free stays at function end after the alloc is delayed.
func.func @_QPwith_free() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFwith_freeEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFwith_freeEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %5 = fir.convert %c1 : (index) -> i32
  %6 = fir.convert %c10 : (index) -> i64
  fir.call @_FortranAAllocatableSetBounds(%4, %5, %6, %6) : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i32, i64, i64) -> ()
  %7 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_FortranAAllocatableSetBounds(!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i32, i64, i64)

// CHECK-LABEL: func.func @_QPwith_free()
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         fir.call @_FortranAAllocatableSetBounds
// CHECK:         cuf.allocate
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 5: Multiple allocatable descriptors are delayed independently.
func.func @_QPmultiple() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFmultipleEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmultipleEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %10 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf64>>> {bindc_name = "b", data_attr = #cuf.cuda<device>, uniq_name = "_QFmultipleEb"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %11 = fir.zero_bits !fir.heap<!fir.array<?xf64>>
  %12 = fir.shape %c0 : (index) -> !fir.shape<1>
  %13 = fir.embox %11(%12) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf64>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf64>>>
  fir.store %13 to %10 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %14 = fir.declare %10 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmultipleEb"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %c42 = arith.constant 42 : index
  %20 = cuf.allocate %14 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {data_attr = #cuf.cuda<device>} -> i32
  %c99 = arith.constant 99 : index
  %21 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %14 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>> {data_attr = #cuf.cuda<device>}
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPmultiple()
// Both allocs move past the constants to just before their first real use.
// "b" is used first (cuf.allocate %b), "a" is used later (cuf.allocate %a).
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 42
// CHECK:         cuf.alloc {{.*}}uniq_name = "_QFmultipleEb"
// CHECK:         fir.store
// CHECK:         fir.declare {{.*}}uniq_name = "_QFmultipleEb"
// CHECK:         cuf.allocate
// CHECK:         arith.constant 99
// CHECK:         cuf.alloc {{.*}}uniq_name = "_QFmultipleEa"
// CHECK:         fir.store
// CHECK:         fir.declare {{.*}}uniq_name = "_QFmultipleEa"
// CHECK:         cuf.allocate
// CHECK:         cuf.free
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 6: Dead descriptor (declare with only cuf.free user) — still moves
// to just before the free.
func.func @_QPdead_var() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFdead_varEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFdead_varEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c42 = arith.constant 42 : index
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPdead_var()
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 42
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 7: First use inside fir.do_loop — cuf.alloc moves before the loop.
func.func @_QPuse_in_loop() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFuse_in_loopEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFuse_in_loopEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  fir.do_loop %i = %c1 to %c10 step %c1 {
    %5 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  }
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPuse_in_loop()
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 99
// CHECK:         arith.constant 1
// CHECK:         arith.constant 10
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         fir.do_loop
// CHECK:           fir.load
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 8: All uses in a successor block (e.g. after a branch from
// stop-in-if). The cuf.alloc should move into that block before the
// first use so it still dominates the successor.
func.func @_QPuse_in_successor(%arg0: !fir.ref<i32>) {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFuse_in_successorEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFuse_in_successorEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c42 = arith.constant 42 : index
  %5 = fir.load %arg0 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %6 = arith.cmpi slt, %5, %c0_i32 : i32
  cf.cond_br %6, ^bb1, ^bb2
^bb1:
  fir.call @_FortranAStopStatementText(%c0_i32, %c0_i32) : (i32, i32) -> none
  fir.unreachable
^bb2:
  %7 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_FortranAStopStatementText(i32, i32) -> none
func.func private @_FortranAAllocatableSetBounds(!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, i32, i64, i64)

// CHECK-LABEL: func.func @_QPuse_in_successor
// All uses are in ^bb2 only, so cuf.alloc+store+declare move directly
// into ^bb2 before the first use.
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 42
// CHECK:         fir.load
// CHECK:         arith.cmpi
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         fir.unreachable
// CHECK:       ^bb2:
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         cuf.allocate
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 9: Uses in multiple successor blocks — cuf.alloc moves to just
// before the entry block's terminator (the branch) so it dominates both.
func.func @_QPuse_in_multi_successor(%arg0: !fir.ref<i32>) {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFuse_in_multi_successorEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFuse_in_multi_successorEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c42 = arith.constant 42 : index
  %5 = fir.load %arg0 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %6 = arith.cmpi slt, %5, %c0_i32 : i32
  cf.cond_br %6, ^bb1, ^bb2
^bb1:
  %7 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
^bb2:
  %8 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPuse_in_multi_successor
// Uses span ^bb1 and ^bb2 — alloc moves to before the entry block's
// terminator (cf.cond_br).
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 42
// CHECK:         fir.load
// CHECK:         arith.constant 0
// CHECK:         arith.cmpi
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         fir.load
// CHECK:         cuf.free
// CHECK:       ^bb2:
// CHECK:         cuf.allocate
// CHECK:         cuf.free

// -----

// Test 10: A host association store sinks with the group, so the allocation is
// still delayed to the first real use.
func.func @_QPhost_assoc() {
  %tuple = fir.alloca tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %slot = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFhost_assocEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFhost_assocEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  fir.store %4 to %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c99 = arith.constant 99 : index
  %5 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  fir.call @_QFPcontained(%tuple) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_QFPcontained(!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>)

// CHECK-LABEL: func.func @_QPhost_assoc
// CHECK:         fir.alloca
// CHECK:         fir.coordinate_of
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         arith.constant 99
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         cuf.allocate
// CHECK:         fir.call @_QFPcontained
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 11: The descriptor is only captured, never used in the host, so the group
// and its store must still land before the call.
func.func @_QPhost_assoc_call() {
  %tuple = fir.alloca tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %slot = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<managed>, uniq_name = "_QFhost_assoc_callEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 3 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<managed>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFhost_assoc_callEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  fir.store %4 to %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QFPgo(%tuple) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<managed>}
  return
}
func.func private @_QFPgo(!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>)

// CHECK-LABEL: func.func @_QPhost_assoc_call
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         fir.call @_QFPgo
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 12: A shared tuple filled in slot order, so the descriptor's own slot is
// defined after an earlier coordinate_of of the same tuple.
func.func @_QPhost_assoc_late_slot() {
  %tuple = fir.alloca tuple<!fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<managed>, uniq_name = "_QFhost_assoc_late_slotEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 3 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<managed>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFhost_assoc_late_slotEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %slot0 = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  %scalar = fir.alloca i32
  fir.store %scalar to %slot0 : !fir.llvm_ptr<!fir.ref<i32>>
  %slot1 = fir.coordinate_of %tuple, %c1_i32 : (!fir.ref<tuple<!fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.store %4 to %slot1 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QFPgo2(%tuple) : (!fir.ref<tuple<!fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<managed>}
  return
}
func.func private @_QFPgo2(!fir.ref<tuple<!fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>)

// CHECK-LABEL: func.func @_QPhost_assoc_late_slot
// CHECK:         fir.coordinate_of {{.*}} -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:         fir.coordinate_of {{.*}} -> !fir.llvm_ptr<!fir.ref<!fir.box
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr<!fir.ref<!fir.box
// CHECK:         fir.call @_QFPgo2
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 13: The tuple is populated in the prologue but the descriptor is not used
// until later, so the group and its store sink past the intervening call.
func.func @_QPhost_assoc_prologue() {
  %tuple = fir.alloca tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %slot = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFhost_assoc_prologueEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFhost_assoc_prologueEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  fir.store %4 to %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QPsetup() : () -> ()
  %5 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  fir.call @_QFPcontained2(%tuple) : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>) -> ()
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_QPsetup()
func.func private @_QFPcontained2(!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>)

// CHECK-LABEL: func.func @_QPhost_assoc_prologue
// CHECK:         fir.coordinate_of
// CHECK:         fir.zero_bits
// CHECK:         fir.embox
// CHECK:         fir.call @_QPsetup
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         cuf.allocate
// CHECK:         fir.call @_QFPcontained2
// CHECK:         cuf.free
// CHECK:         return

// -----

// Test 14: Uses span several successor blocks, but ^bb2 dominates them all, so
// the group sinks into ^bb2 instead of stopping at the entry terminator.
func.func @_QPuse_in_dominating_successor(%arg0: !fir.ref<i32>) {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFuse_in_dominating_successorEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFuse_in_dominating_successorEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c0_i32 = arith.constant 0 : i32
  %5 = fir.load %arg0 : !fir.ref<i32>
  %6 = arith.cmpi slt, %5, %c0_i32 : i32
  cf.cond_br %6, ^bb1, ^bb2
^bb1:
  fir.call @_FortranAStopStatementText(%c0_i32, %c0_i32) : (i32, i32) -> none
  fir.unreachable
^bb2:
  fir.call @_QPinit_runtime() : () -> ()
  %7 = cuf.allocate %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  %8 = fir.load %arg0 : !fir.ref<i32>
  %9 = arith.cmpi slt, %8, %c0_i32 : i32
  cf.cond_br %9, ^bb3, ^bb4
^bb3:
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
^bb4:
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_FortranAStopStatementText(i32, i32) -> none
func.func private @_QPinit_runtime()

// CHECK-LABEL: func.func @_QPuse_in_dominating_successor
// CHECK:         cf.cond_br
// CHECK:       ^bb1:
// CHECK:         fir.unreachable
// CHECK:       ^bb2:
// The allocation happens after the call, not before the branch.
// CHECK:         fir.call @_QPinit_runtime
// CHECK:         cuf.alloc
// CHECK:         fir.store
// CHECK:         fir.declare
// CHECK:         cuf.allocate

// -----

// Test 15: A store into an !fir.llvm_ptr that is not a tuple slot goes to an
// unknown destination, so it is an ordinary use and pins the group.
func.func @_QPllvm_ptr_not_tuple(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFllvm_ptr_not_tupleEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFllvm_ptr_not_tupleEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %opaque = fir.convert %arg0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.store %4 to %opaque : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QPsetup3() : () -> ()
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_QPsetup3()

// CHECK-LABEL: func.func @_QPllvm_ptr_not_tuple
// CHECK:         fir.convert
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         fir.call @_QPsetup3
// CHECK:         cuf.free

// -----

// Test 16: The tuple slot is read back in the host, so the group sinks past the
// call. A coordinate_of only computes the slot address, so the group stays
// before the fir.load that consumes it.
func.func @_QPtuple_slot_read() {
  %tuple = fir.alloca tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %slot = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFtuple_slot_readEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtuple_slot_readEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  fir.store %4 to %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QPsetup4() : () -> ()
  %reread = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %5 = fir.load %reread : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %6 = cuf.allocate %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_QPsetup4()

// CHECK-LABEL: func.func @_QPtuple_slot_read
// CHECK:         fir.coordinate_of
// CHECK:         fir.embox
// CHECK:         fir.call @_QPsetup4
// CHECK:         fir.coordinate_of
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         fir.load
// CHECK:         cuf.allocate
// CHECK:         cuf.free

// -----

// Test 17: The same coordinate_of feeds both the host-association capture store
// and a later load (as CSE can produce). The coordinate_of is not a real use.
func.func @_QPshared_slot_addr() {
  %tuple = fir.alloca tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %c0_i32 = arith.constant 0 : i32
  %slot = fir.coordinate_of %tuple, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFshared_slot_addrEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
  %c0 = arith.constant 0 : index
  %2 = fir.shape %c0 : (index) -> !fir.shape<1>
  %3 = fir.embox %1(%2) {allocator_idx = 2 : i32} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4 = fir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFshared_slot_addrEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  fir.store %4 to %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  fir.call @_QPsetup5() : () -> ()
  %5 = fir.load %slot : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>>
  %6 = cuf.allocate %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}
func.func private @_QPsetup5()

// CHECK-LABEL: func.func @_QPshared_slot_addr
// CHECK:         fir.coordinate_of
// CHECK:         fir.embox
// CHECK:         fir.call @_QPsetup5
// CHECK:         cuf.alloc
// CHECK:         fir.store {{.*}} : !fir.ref<!fir.box
// CHECK:         fir.declare
// CHECK:         fir.store {{.*}} : !fir.llvm_ptr
// CHECK:         fir.load
// CHECK:         cuf.allocate
// CHECK:         cuf.free
