// Test routing decisions for _FortranAAssignSimple vs _FortranAAssign in
// hlfir.assign lowering to FIR. The routing logic lives in ConvertToFIR.cpp
// and has two main decision points:
//
//   1. Allocatable path (isAllocatableAssignment): routes to AssignSimple when
//      isa_trivial(elementType) && ranksMatch && !volatile
//
//   2. Non-allocatable array path (lhs.isArray()): routes to AssignSimple when
//      !polymorphic && isa_trivial(elementType) && ranksMatch && !volatile
//
// This file verifies every boundary condition of those routing decisions.

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

// ---------------------------------------------------------------------------
// POSITIVE CASES: should route to _FortranAAssignSimple
// ---------------------------------------------------------------------------

// Allocatable rank-1 i32 array: trivial integer type, ranks match (1==1),
// non-volatile.  This is the canonical "simple allocatable" case.
func.func @alloc_i32_array(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
                            %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xi32>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL: func.func @alloc_i32_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Allocatable rank-1 f64 array: trivial real type, ranks match (1==1),
// non-volatile.  Verifies that real (double-precision) types are routed simply.
func.func @alloc_f64_array(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>,
                            %arg1: !fir.box<!fir.array<?xf64>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xf64>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  return
}
// CHECK-LABEL: func.func @alloc_f64_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Allocatable rank-1 complex<f32> array: trivial complex type, ranks match,
// non-volatile.  Verifies complex intrinsic types (MLIR ComplexType) are
// routed simply — isa_complex() checks mlir::ComplexType with float element.
func.func @alloc_complex_array(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>,
                                %arg1: !fir.box<!fir.array<?xcomplex<f32>>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xcomplex<f32>>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
  return
}
// CHECK-LABEL: func.func @alloc_complex_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Allocatable rank-1 logical<4> array: trivial logical type, ranks match,
// non-volatile.  Verifies logical intrinsic types are routed simply.
func.func @alloc_logical_array(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>,
                                %arg1: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?x!fir.logical<4>>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  return
}
// CHECK-LABEL: func.func @alloc_logical_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Allocatable rank-2 i32 array: trivial integer type, ranks match (2==2),
// non-volatile.  Verifies that the rank-match check works for rank-2.
func.func @alloc_i32_2d(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>,
                         %arg1: !fir.box<!fir.array<?x?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?x?xi32>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  return
}
// CHECK-LABEL: func.func @alloc_i32_2d(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Non-allocatable rank-1 i32 array: non-polymorphic, trivial integer, ranks
// match (1==1), non-volatile.  Canonical "simple non-allocatable array" case.
func.func @nonalloc_i32_array(%arg0: !fir.box<!fir.array<?xi32>>,
                               %arg1: !fir.ref<!fir.array<100xi32>>) {
  %c100 = arith.constant 100 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
  hlfir.assign %2#0 to %0#0 : !fir.ref<!fir.array<100xi32>>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL: func.func @nonalloc_i32_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Non-allocatable rank-1 f32 array: non-polymorphic, trivial real, ranks
// match (1==1), non-volatile.  Verifies f32 (single-precision) is routed simply.
func.func @nonalloc_f32_array(%arg0: !fir.box<!fir.array<?xf32>>,
                               %arg1: !fir.ref<!fir.array<100xf32>>) {
  %c100 = arith.constant 100 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "y"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  hlfir.assign %2#0 to %0#0 : !fir.ref<!fir.array<100xf32>>, !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL: func.func @nonalloc_f32_array(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------

// Non-allocatable rank-1 logical<4> array: non-polymorphic, trivial logical,
// ranks match (1==1), non-volatile.  Verifies logical types in the non-alloc path.
func.func @nonalloc_logical(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>>,
                             %arg1: !fir.ref<!fir.array<100x!fir.logical<4>>>) {
  %c100 = arith.constant 100 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "y"} : (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.ref<!fir.array<100x!fir.logical<4>>>)
  hlfir.assign %2#0 to %0#0 : !fir.ref<!fir.array<100x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>
  return
}
// CHECK-LABEL: func.func @nonalloc_logical(
// CHECK:       fir.call @_FortranAAssignSimple(
// CHECK-NOT:   fir.call @_FortranAAssign(

// ---------------------------------------------------------------------------
// NEGATIVE CASES: should NOT use AssignSimple, route to _FortranAAssign
// ---------------------------------------------------------------------------

// Allocatable derived type array: !isa_trivial(elementType).  Derived types
// may need finalization/copy semantics so they cannot use the simple path.
func.func @alloc_derived(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<t1{x:f32}>>>>>,
                          %arg1: !fir.box<!fir.array<?x!fir.type<t1{x:f32}>>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?x!fir.type<t1{x:f32}>>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<t1{x:f32}>>>>>
  return
}
// CHECK-LABEL: func.func @alloc_derived(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Allocatable volatile i32 array: volatile flag is set on the ref itself.
// The volatile is on !fir.ref<..., volatile>, so isa_volatile_type(lhs.getType())
// returns true and AssignSimple cannot be used.
func.func @alloc_volatile(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>, volatile>,
                           %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xi32>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>, volatile>
  return
}
// CHECK-LABEL: func.func @alloc_volatile(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Allocatable scalar-to-array: rank mismatch (RHS rank 0, LHS rank 1).
// Even though the element type is trivial (i32), the rank mismatch (scalar
// RHS to array LHS) violates the lhs.getRank() == rhs.getRank() precondition
// in the allocatable path, so this must route to the full _FortranAAssign
// path to handle scalar broadcasting.
func.func @alloc_scalar_to_array(%lhs: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
                                  %rhs: i32) {
  hlfir.assign %rhs to %lhs realloc : i32,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL: func.func @alloc_scalar_to_array(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Non-allocatable scalar-to-array: rank mismatch (RHS rank 0, LHS rank 1).
// Broadcasting a scalar across an array requires the full _FortranAAssign path.
func.func @nonalloc_scalar_to_array(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: i32) {
  hlfir.assign %rhs to %lhs : i32, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL: func.func @nonalloc_scalar_to_array(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Non-allocatable polymorphic array: lhs.isPolymorphic() is true.  Polymorphic
// assignments must be handled by _FortranAAssign for correct dynamic type
// semantics; AssignSimple is excluded by the !lhs.isPolymorphic() guard.
func.func @nonalloc_polymorphic(%lhs: !fir.class<!fir.array<?x!fir.type<t2{y:i64}>>>,
                                 %rhs: !fir.class<!fir.array<?x!fir.type<t2{y:i64}>>>) {
  hlfir.assign %rhs to %lhs : !fir.class<!fir.array<?x!fir.type<t2{y:i64}>>>,
      !fir.class<!fir.array<?x!fir.type<t2{y:i64}>>>
  return
}
// CHECK-LABEL: func.func @nonalloc_polymorphic(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Non-allocatable volatile array: volatile flag prevents the simple path.
// Volatile arrays need proper memory-ordering semantics.
func.func @nonalloc_volatile_array(%lhs: !fir.box<!fir.array<?xi32>, volatile>,
                                    %rhs: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %rhs to %lhs : !fir.box<!fir.array<?xi32>>,
      !fir.box<!fir.array<?xi32>, volatile>
  return
}
// CHECK-LABEL: func.func @nonalloc_volatile_array(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Non-allocatable derived type array: !isa_trivial(elementType).  Derived
// types in the non-allocatable array path also require the full assign path.
func.func @nonalloc_derived_array(%lhs: !fir.box<!fir.array<?x!fir.type<t3{z:f32}>>>,
                                   %rhs: !fir.box<!fir.array<?x!fir.type<t3{z:f32}>>>) {
  hlfir.assign %rhs to %lhs : !fir.box<!fir.array<?x!fir.type<t3{z:f32}>>>,
      !fir.box<!fir.array<?x!fir.type<t3{z:f32}>>>
  return
}
// CHECK-LABEL: func.func @nonalloc_derived_array(
// CHECK:       fir.call @_FortranAAssign(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------
// OTHER ASSIGN VARIANTS: tests for other runtime entry points
// ---------------------------------------------------------------------------

// Allocatable with temporary_lhs: routes to _FortranAAssignTemporary.
// The temporary_lhs attribute indicates the LHS is a compiler-generated temp,
// so reallocation must not finalize it; AssignTemporary handles this correctly.
func.func @alloc_temporary(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
                            %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc temporary_lhs : !fir.box<!fir.array<?xi32>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL: func.func @alloc_temporary(
// CHECK:       fir.call @_FortranAAssignTemporary(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Allocatable polymorphic LHS: routes to _FortranAAssignPolymorphic.
// Polymorphic allocatable assignments require setting the LHS dynamic type
// to match the RHS, which only AssignPolymorphic does correctly.
func.func @alloc_polymorphic(%lhs: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t4{w:i32}>>>>>,
                              %rhs: !fir.class<!fir.array<?x!fir.type<t4{w:i32}>>>) {
  hlfir.assign %rhs to %lhs realloc : !fir.class<!fir.array<?x!fir.type<t4{w:i32}>>>,
      !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t4{w:i32}>>>>>
  return
}
// CHECK-LABEL: func.func @alloc_polymorphic(
// CHECK:       fir.call @_FortranAAssignPolymorphic(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(

// ---------------------------------------------------------------------------

// Allocatable with keep_lhs_len: routes to _FortranAAssignExplicitLengthCharacter.
// The keep_lhs_length_if_realloc attribute means the LHS character length must
// not change on reallocation; this requires the dedicated character assign path.
func.func @alloc_explicit_char(%lhs: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>,
                                %rhs: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  hlfir.assign %rhs to %lhs realloc keep_lhs_len : !fir.box<!fir.array<?x!fir.char<1,?>>>,
      !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  return
}
// CHECK-LABEL: func.func @alloc_explicit_char(
// CHECK:       fir.call @_FortranAAssignExplicitLengthCharacter(
// CHECK-NOT:   fir.call @_FortranAAssignSimple(
