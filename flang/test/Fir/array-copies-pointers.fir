// Test array-copy-value pass (copy elision) with array assignment
// involving Fortran pointers. Focus in only on wether copy ellision
// is made or not.
// RUN: fir-opt %s --array-value-copy -split-input-file | FileCheck --check-prefixes=ALL,NOOPT %s
// RUN: fir-opt %s --array-value-copy="optimize-conflicts=true" -split-input-file | FileCheck --check-prefixes=ALL,OPT %s

// Test `pointer(:) = array(:)`
// ALL-LABEL: func @maybe_overlap
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
func.func @maybe_overlap(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1 : !fir.ref<!fir.array<100xf32>> {fir.target}) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %3, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// -----

// Test `pointer(:) = array(:)`
// ALL-LABEL: func @no_overlap1
// OPT-NOT: fir.allocmem
// NOOPT: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
func.func @no_overlap1(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1 : !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %3, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %4 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// -----

// Test `pointer(:) = pointer(:)`
// ALL-LABEL: func @no_overlap
// ALL-NOT: fir.allocmem
// ALL:     fir.do_loop
// ALL:       fir.array_coor
// ALL:       fir.array_coor
// ALL:       fir.store
func.func @no_overlap(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<100xf32>) {
    %4 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %5 = fir.array_update %arg3, %4, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %5 : !fir.array<100xf32>
  }
  fir.array_merge_store %2, %3 to %arg0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// -----

// Test `array(:) = pointer(:)`
// ALL-LABEL: func @maybe_overlap_2
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
func.func @maybe_overlap_2(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>> {fir.target}) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %arg1 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

// -----

// Test `array(:) = pointer(:)`
// ALL-LABEL: func @no_overlap_2
// OPT-NOT: fir.allocmem
// NOOPT: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
func.func @no_overlap_2(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %arg1 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

// -----

// Test `pointer1(:) = pointer2(:)`
// ALL-LABEL: func @maybe_overlap_3
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
func.func @maybe_overlap_3(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ptr<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.alloca f32
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %arg1(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %arg1 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ptr<!fir.array<100xf32>>
  return
}

// -----

// Test derived_target(:)%i = integer_pointer(:)
// The integer pointer may be aliasing the derived target component.
// ALL-LABEL: func @derived_whose_component_may_be_aliased
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<4x!fir.type<some_type{i:i32}>>
// ALL-COUNT-3: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<4x!fir.type<some_type{i:i32}>>>
func.func @derived_whose_component_may_be_aliased(%arg0: !fir.box<!fir.array<4x!fir.type<some_type{i:i32}>>> {fir.target}, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
  %c4 = arith.constant 4 : index
  %0 = fir.field_index i, !fir.type<some_type{i:i32}>
  %c1 = arith.constant 1 : index
  %1 = fir.slice %c1, %c4, %c1 path %0 : (index, index, index, !fir.field) -> !fir.slice<1>
  %2 = fir.array_load %arg0 [%1] : (!fir.box<!fir.array<4x!fir.type<some_type{i:i32}>>>, !fir.slice<1>) -> !fir.array<4xi32>
  %3 = fir.load %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
  %5 = fir.shift %4#0 : (index) -> !fir.shift<1>
  %6 = fir.array_load %3(%5) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.array<?xi32>
  %7 = arith.subi %c4, %c1 : index
  %8 = fir.do_loop %arg2 = %c0 to %7 step %c1 unordered iter_args(%arg3 = %2) -> (!fir.array<4xi32>) {
    %9 = fir.array_fetch %6, %arg2 : (!fir.array<?xi32>, index) -> i32
    %10 = fir.array_update %arg3, %9, %arg2 : (!fir.array<4xi32>, i32, index) -> !fir.array<4xi32>
    fir.result %10 : !fir.array<4xi32>
  }
  fir.array_merge_store %2, %8 to %arg0[%1] : !fir.array<4xi32>, !fir.array<4xi32>, !fir.box<!fir.array<4x!fir.type<some_type{i:i32}>>>, !fir.slice<1>
  return
}

// -----

// Test real_target = complex_target(:)%re
// The real pointer may be aliasing the complex real part.
// ALL-LABEL: func @complex_real_aliasing
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>
// ALL-COUNT-3: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xf32>>
func.func @complex_real_aliasing(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %arg1: !fir.ref<!fir.array<4x!fir.complex<4>>> {fir.target}) {
  %c4 = arith.constant 4 : index
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %c0 = arith.constant 0 : index
  %1:3 = fir.box_dims %0, %c0 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  %2 = fir.shift %1#0 : (index) -> !fir.shift<1>
  %3 = fir.array_load %0(%2) : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.array<?xf32>
  %c0_i32 = arith.constant 0 : i32
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %c1 = arith.constant 1 : index
  %5 = fir.slice %c1, %c4, %c1 path %c0_i32 : (index, index, index, i32) -> !fir.slice<1>
  %6 = fir.array_load %arg1(%4) [%5] : (!fir.ref<!fir.array<4x!fir.complex<4>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<4xf32>
  %7 = arith.subi %c4, %c1 : index
  %8 = fir.do_loop %arg2 = %c0 to %7 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<?xf32>) {
    %9 = fir.array_fetch %6, %arg2 : (!fir.array<4xf32>, index) -> f32
    %10 = fir.array_update %arg3, %9, %arg2 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
    fir.result %10 : !fir.array<?xf32>
  }
  fir.array_merge_store %3, %8 to %0 : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.ptr<!fir.array<?xf32>>>
  return
}

// -----

// Test `array(:) = pointer(:)`
// ALL-LABEL: func @maybe_overlap_3
// ALL: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.do_loop
// ALL: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
fir.global @_QMdataEglob target : !fir.array<100xf32> {
  %0 = fir.undefined !fir.array<100xf32>
  fir.has_value %0 : !fir.array<100xf32>
}

func.func @maybe_overlap_3(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>> {fir.target}) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMdataEglob) : !fir.ref<!fir.array<100xf32>>
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}

// -----

// Test `array(:) = pointer(:)`
// ALL-LABEL: func @no_overlap_3
// OPT-NOT: fir.allocmem
// NOOPT: %[[ALLOC:.*]] = fir.allocmem !fir.array<100xf32>
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.do_loop
// NOOPT: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<100xf32>>
fir.global @_QMdataEglob : !fir.array<100xf32> {
  %0 = fir.undefined !fir.array<100xf32>
  fir.has_value %0 : !fir.array<100xf32>
}

func.func @no_overlap_3(%arg0: !fir.ptr<!fir.array<100xf32>>, %arg1: !fir.ref<!fir.array<100xf32>> {fir.target}) {
  %c100 = arith.constant 100 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QMdataEglob) : !fir.ref<!fir.array<100xf32>>
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ptr<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %3 = fir.array_load %0(%1) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.array<100xf32>
  %4 = fir.do_loop %arg2 = %c0 to %c99 step %c1 unordered iter_args(%arg3 = %3) -> (!fir.array<100xf32>) {
    %5 = fir.array_fetch %2, %arg2 : (!fir.array<100xf32>, index) -> f32
    %6 = fir.array_update %arg3, %5, %arg2 : (!fir.array<100xf32>, f32, index) -> !fir.array<100xf32>
    fir.result %6 : !fir.array<100xf32>
  }
  fir.array_merge_store %3, %4 to %0 : !fir.array<100xf32>, !fir.array<100xf32>, !fir.ref<!fir.array<100xf32>>
  return
}
