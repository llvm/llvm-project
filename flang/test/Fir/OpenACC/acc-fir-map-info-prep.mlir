// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// Nested box member copyin: map_info carries varPtrPtr = descriptor slot.

// CHECK-LABEL: func.func @nested_box_member
// CHECK: %[[BOX:.*]] = fir.load %[[SLOT:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
// CHECK: acc.map_info varPtr(%[[ADDR]] : !fir.heap<!fir.array<?xf32>>)
// CHECK-SAME: varPtrPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK-SAME: desc(%[[BOX]] : !fir.box<!fir.heap<!fir.array<?xf32>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK-NOT: acc.copyin
// CHECK: acc.data

func.func @nested_box_member() {
  %0 = fir.undefined !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1 = fir.load %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %2 = fir.box_addr %1 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %copy = acc.copyin varPtr(%2 : !fir.heap<!fir.array<?xf32>>)
      dataClause(acc_copyin) structured(true) name("m") -> !fir.heap<!fir.array<?xf32>>
  acc.data dataOperands(%copy : !fir.heap<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @derived_with_box
// CHECK: %[[VAR:.*]] = fir.undefined !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
// CHECK: fir.type_desc !fir.type<_QMtypesTderived{{.*}}>
// CHECK: %[[TDESC:.*]] = fir.address_of(@_QMtypesEXdtXderived)
// CHECK: fir.field_index sizeinbytes
// CHECK: %[[ADDR:.*]] = fir.coordinate_of %[[TDESC]], sizeinbytes
// CHECK: %[[SIZE:.*]] = fir.load %[[ADDR]]
// CHECK: acc.map_info varPtr(%[[VAR]] : !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: acc.copyin
// CHECK: acc.data

fir.global linkonce_odr @_QMtypesEXdtXderived constant target : !fir.type<_QM__fortran_type_infoTderivedtype{sizeinbytes:i64}> {
  %0 = fir.undefined !fir.type<_QM__fortran_type_infoTderivedtype{sizeinbytes:i64}>
  fir.has_value %0 : !fir.type<_QM__fortran_type_infoTderivedtype{sizeinbytes:i64}>
}

func.func @derived_with_box() {
  %0 = fir.undefined !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  %copy = acc.copyin varPtr(%0 : !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)
      dataClause(acc_copyin) name("st")
      -> !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  acc.data dataOperands(%copy : !fir.ref<!fir.type<_QMtypesTderived{member:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) {
    acc.terminator
  }
  return
}

// firstprivate_map is a live-in (not on dataOperands) but still gets map_info.
// A partial array section keeps the full-array byte size on map_info; bounds
// carry the section.

// CHECK-LABEL: func.func @firstprivate_partial_array
// CHECK: %[[SOURCE_EXTENT:.*]] = arith.constant 100 : index
// CHECK: %[[BOUND:.*]] = acc.bounds
// CHECK-SAME: sourceExtent(%[[SOURCE_EXTENT]] : index)
// CHECK: %[[SIZE:.*]] = arith.constant 400 : i64
// CHECK: acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<100xf32>>)
// CHECK-SAME: bounds(%[[BOUND]])
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(to,private)
// CHECK-NOT: acc.firstprivate_map
func.func @firstprivate_partial_array(%w: !fir.ref<!fir.array<100xf32>>) {
  %c1 = arith.constant 1 : index
  %lb = arith.constant 4 : index
  %ub = arith.constant 7 : index
  %ext = arith.constant 100 : index
  %st = arith.constant 1 : index
  %bnd = acc.bounds lowerbound(%lb : index) upperbound(%ub : index)
      extent(%ext : index) stride(%st : index) startIdx(%c1 : index)
      sourceExtent(%ext : index)
  %fp = acc.firstprivate_map varPtr(%w : !fir.ref<!fir.array<100xf32>>)
      bounds(%bnd) name("w") -> !fir.ref<!fir.array<100xf32>>
  return
}
