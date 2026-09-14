// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// When a data region maps both a descriptor entry and an implicit present of the
// pointee address, the implicit sibling must not inherit CFI or attach facts
// from the descriptor map. Only the explicit descriptor entry keeps
// descKind(cfi) and ptr_and_obj.

// CHECK-LABEL: func.func @assumed_shape_with_implicit_present
// CHECK: %[[BOX:.*]] = fir.undefined !fir.box<!fir.array<?xf32>>
// CHECK: %[[DESC:.*]] = acc.map_info var(%[[BOX]] : !fir.box<!fir.array<?xf32>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to)
// CHECK: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[BASE:.*]] = acc.map_info varPtr(%[[ADDR]] : !fir.ref<!fir.array<?xf32>>)
// CHECK-SAME: size(%[[ZERO]] : i64)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(implicit,present)
// CHECK-NOT: ptr_and_obj
// CHECK: acc.data dataOperands(%[[DESC]], %[[BASE]]
func.func @assumed_shape_with_implicit_present() {
  %box = fir.undefined !fir.box<!fir.array<?xf32>>
  %desc = acc.copyin var(%box : !fir.box<!fir.array<?xf32>>)
      name("a") -> !fir.box<!fir.array<?xf32>>
  %addr = fir.box_addr %box : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %base = acc.present varPtr(%addr : !fir.ref<!fir.array<?xf32>>)
      implicit(true) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%desc, %base : !fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @allocatable_array_with_implicit_present
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: %[[DESC:.*]] = acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: %[[LOAD:.*]] = fir.load %[[SLOT]]
// CHECK: %[[BADDR:.*]] = fir.box_addr %[[LOAD]]
// CHECK: %[[CONV:.*]] = fir.convert %[[BADDR]]
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[BASE:.*]] = acc.map_info varPtr(%[[CONV]] : !fir.ref<!fir.array<?xf32>>)
// CHECK-SAME: size(%[[ZERO]] : i64)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(implicit,present)
// CHECK-NOT: varPtrPtr
// CHECK: acc.data dataOperands(%[[DESC]], %[[BASE]]
func.func @allocatable_array_with_implicit_present() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %desc = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
      name("a") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %load = fir.load %slot : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %baddr = fir.box_addr %load : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %conv = fir.convert %baddr : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %base = acc.present varPtr(%conv : !fir.ref<!fir.array<?xf32>>)
      implicit(true) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%desc, %base : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @pointer_array_with_implicit_present
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: %[[DESC:.*]] = acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: %[[LOAD:.*]] = fir.load %[[SLOT]]
// CHECK: %[[BADDR:.*]] = fir.box_addr %[[LOAD]]
// CHECK: %[[CONV:.*]] = fir.convert %[[BADDR]]
// CHECK: %[[BASE:.*]] = acc.map_info varPtr(%[[CONV]] : !fir.ref<!fir.array<?xf32>>)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(implicit,present)
// CHECK: acc.data dataOperands(%[[DESC]], %[[BASE]]
func.func @pointer_array_with_implicit_present() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %desc = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      name("a") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %load = fir.load %slot : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %baddr = fir.box_addr %load : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  %conv = fir.convert %baddr : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %base = acc.present varPtr(%conv : !fir.ref<!fir.array<?xf32>>)
      implicit(true) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%desc, %base : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// Struct members with both descriptor and implicit pointee present entries.
// Scalar c_ptr, allocatable scalar, and polymorphic cases are covered in
// acc-fir-map-info-prep-types.mlir.
// CHECK-LABEL: func.func @struct_array_members_with_implicit_present
// CHECK: %[[ALLOC_ARR:.*]] = fir.coordinate_of %{{.*}}, alloc_arr
// CHECK: %[[PTR_ARR:.*]] = fir.coordinate_of %{{.*}}, ptr_arr
// CHECK: %[[DESC0:.*]] = acc.map_info varPtr(%[[ALLOC_ARR]] : {{[^)]*}})
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: %[[DESC1:.*]] = acc.map_info varPtr(%[[PTR_ARR]] : {{[^)]*}})
// CHECK-SAME: elementSize(4)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: %[[BASE0:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(implicit,present)
// CHECK: %[[BASE1:.*]] = acc.map_info varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(implicit,present)
// CHECK: acc.data dataOperands(%[[DESC0]], %[[DESC1]], %[[BASE0]], %[[BASE1]]
func.func @struct_array_members_with_implicit_present() {
  %h = fir.undefined !fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  %alloc_arr = fir.coordinate_of %h, alloc_arr : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %ptr_arr = fir.coordinate_of %h, ptr_arr : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %d0 = acc.copyin varPtr(%alloc_arr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
      name("h%alloc_arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %d1 = acc.copyin varPtr(%ptr_arr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      name("h%ptr_arr") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %l0 = fir.load %alloc_arr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %a0 = fir.box_addr %l0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  %c0 = fir.convert %a0 : (!fir.heap<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %b0 = acc.present varPtr(%c0 : !fir.ref<!fir.array<?xf32>>)
      implicit(true) name("h%alloc_arr") -> !fir.ref<!fir.array<?xf32>>
  %l1 = fir.load %ptr_arr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %a1 = fir.box_addr %l1 : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  %c1 = fir.convert %a1 : (!fir.ptr<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %b1 = acc.present varPtr(%c1 : !fir.ref<!fir.array<?xf32>>)
      implicit(true) name("h%ptr_arr") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%d0, %d1, %b0, %b1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}
