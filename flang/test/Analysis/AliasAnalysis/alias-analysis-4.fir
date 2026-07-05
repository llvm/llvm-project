// check that hlfir.designate can be followed by alias analysis

// use --mlir-disable-threading so that the AA queries are serialised
// as well as its diagnostic output.
// RUN: fir-opt %s --test-fir-alias-analysis -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// designate for a derived type component:
// module m
//   type t
//     real :: array(42)
//   end type t
//   type (t) :: glbl
// contains
//   subroutine test(arg)
//     real :: arg(42)
//     glbl%array = arg
//   end subroutine test
// end module m

// A global can't alias with a dummy argument
// CHECK: arg#0 <-> glbl%array#0: NoAlias

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.target_triple = "aarch64-unknown-linux-gnu"} {
  fir.global @_QMmEglbl : !fir.type<_QMmTt{array:!fir.array<42xf32>}> {
    %0 = fir.undefined !fir.type<_QMmTt{array:!fir.array<42xf32>}>
    fir.has_value %0 : !fir.type<_QMmTt{array:!fir.array<42xf32>}>
  }
  func.func @_QMmPtest(%arg0: !fir.ref<!fir.array<42xf32>> {fir.bindc_name = "arg"}) {
    %c42 = arith.constant 42 : index
    %0 = fir.address_of(@_QMmEglbl) : !fir.ref<!fir.type<_QMmTt{array:!fir.array<42xf32>}>>
    %1:2 = hlfir.declare %0 {uniq_name = "_QMmEglbl"} : (!fir.ref<!fir.type<_QMmTt{array:!fir.array<42xf32>}>>) -> (!fir.ref<!fir.type<_QMmTt{array:!fir.array<42xf32>}>>, !fir.ref<!fir.type<_QMmTt{array:!fir.array<42xf32>}>>)
    %2 = fir.shape %c42 : (index) -> !fir.shape<1>
    %3:2 = hlfir.declare %arg0(%2) {uniq_name = "_QMmFtestEarg", test.ptr = "arg"} : (!fir.ref<!fir.array<42xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xf32>>, !fir.ref<!fir.array<42xf32>>)
    %4 = hlfir.designate %1#0{"array"} shape %2 {test.ptr = "glbl%array"} : (!fir.ref<!fir.type<_QMmTt{array:!fir.array<42xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<42xf32>>
    hlfir.assign %3#0 to %4 : !fir.ref<!fir.array<42xf32>>, !fir.ref<!fir.array<42xf32>>
    return
  }
}

// -----

// designate for an array element

// two dummy arguments don't alias
// CHECK: array0#0 <-> array1#0: NoAlias

func.func @array_element(%arg0: !fir.ref<!fir.array<2x42xi32>>, %arg1: !fir.ref<!fir.array<2x42xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c41 = arith.constant 41 : index
    %c42 = arith.constant 42 : index
    %shape = fir.shape %c42 : (index) -> !fir.shape<1>
    %array0 = hlfir.designate %arg0 (%c0, %c0:%c41:%c1) shape %shape {test.ptr = "array0"} : (!fir.ref<!fir.array<2x42xi32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    %array1 = hlfir.designate %arg1 (%c1, %c0:%c41:%c1) shape %shape {test.ptr = "array1"} : (!fir.ref<!fir.array<2x42xi32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    return
}

// -----

// FIXME: designate doesn't understand non-overlappning array indices
// make sure that we say MayAlias and not MustAlias until array indexes are understood
// CHECK: array2#0 <-> array3#0: MayAlias

func.func @array_element_same_source(%arg0: !fir.ref<!fir.array<2x42xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c41 = arith.constant 41 : index
    %c42 = arith.constant 42 : index
    %shape = fir.shape %c42 : (index) -> !fir.shape<1>
    %array2 = hlfir.designate %arg0 (%c0, %c0:%c41:%c1) shape %shape {test.ptr = "array2"} : (!fir.ref<!fir.array<2x42xi32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    %array3 = hlfir.designate %arg0 (%c1, %c0:%c41:%c1) shape %shape {test.ptr = "array3"} : (!fir.ref<!fir.array<2x42xi32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    return
}