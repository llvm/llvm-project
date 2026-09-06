// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s

// Descriptor-backed variables mapped through their descriptor or c_ptr slot,
// without a sibling implicit present of the pointee. When var is the
// descriptor, desc is omitted from map_info whenever descKind names CFI on var.

// CHECK-LABEL: func.func @assumed_shape
// CHECK: %[[BOX:.*]] = fir.undefined !fir.box<!fir.array<?xf32>>
// CHECK: acc.map_info var(%[[BOX]] : !fir.box<!fir.array<?xf32>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: ptr_and_obj
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @assumed_shape() {
  %box = fir.undefined !fir.box<!fir.array<?xf32>>
  %copy = acc.copyin var(%box : !fir.box<!fir.array<?xf32>>)
      name("a") -> !fir.box<!fir.array<?xf32>>
  acc.data dataOperands(%copy : !fir.box<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @allocatable_array
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @allocatable_array() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %copy = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
      name("a") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  acc.data dataOperands(%copy : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @pointer_array
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @pointer_array() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %copy = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      name("a") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  acc.data dataOperands(%copy : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @allocatable_integer
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,from,ptr_and_obj)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @allocatable_integer() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.heap<i32>>>
  %copy = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.heap<i32>>>)
      dataClause(acc_copy) name("n")
      -> !fir.ref<!fir.box<!fir.heap<i32>>>
  acc.copyout accPtr(%copy : !fir.ref<!fir.box<!fir.heap<i32>>>)
      to varPtr(%slot : !fir.ref<!fir.box<!fir.heap<i32>>>)
      dataClause(acc_copy) name("n")
  acc.data dataOperands(%copy : !fir.ref<!fir.box<!fir.heap<i32>>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @polymorphic_entity
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @polymorphic_entity() {
  %slot = fir.undefined !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>
  %copy = acc.copyin varPtr(%slot : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>)
      name("p") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>
  acc.data dataOperands(%copy : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>) {
    acc.terminator
  }
  return
}

// A c_ptr mapped by itself transfers only the 8-byte address object; its
// pointee is not mapped, so there is nothing to attach.
// CHECK-LABEL: func.func @cptr_copyin
// CHECK: %[[CPTR:.*]] = fir.undefined !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
// CHECK: acc.map_info varPtr(%[[CPTR]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
// CHECK-SAME: elementSize(8)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @cptr_copyin() {
  %p = fir.undefined !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  %copy = acc.copyin varPtr(%p : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
      name("p") -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  acc.data dataOperands(%copy : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
    acc.terminator
  }
  return
}

// CHECK-LABEL: func.func @attach_pointer
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(ptr_and_obj)
// CHECK-NOT: acc.attach
// CHECK: acc.data
func.func @attach_pointer() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %att = acc.attach varPtr(%slot : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      name("a") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  acc.data dataOperands(%att : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
    acc.terminator
  }
  return
}

// Derived-type members: same attach / CFI facts as the standalone entities.
// CHECK-LABEL: func.func @struct_members
// CHECK-DAG: %[[ALLOC_ARR:.*]] = fir.coordinate_of %{{.*}}, alloc_arr
// CHECK-DAG: %[[PTR_ARR:.*]] = fir.coordinate_of %{{.*}}, ptr_arr
// CHECK-DAG: %[[ALLOC_INT:.*]] = fir.coordinate_of %{{.*}}, alloc_int
// CHECK-DAG: %[[POLY:.*]] = fir.coordinate_of %{{.*}}, poly
// CHECK-DAG: %[[CP:.*]] = fir.coordinate_of %{{.*}}, cp
// CHECK: acc.map_info varPtr(%[[ALLOC_ARR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: acc.map_info varPtr(%[[PTR_ARR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: acc.map_info varPtr(%[[ALLOC_INT]] : !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,from,ptr_and_obj)
// CHECK: acc.map_info varPtr(%[[POLY]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// CHECK: acc.map_info varPtr(%[[CP]] : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
// CHECK-SAME: elementSize(8)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: acc.copyin
// CHECK: acc.data
func.func @struct_members() {
  %h = fir.undefined !fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>
  %alloc_arr = fir.coordinate_of %h, alloc_arr : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %ptr_arr = fir.coordinate_of %h, ptr_arr : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %alloc_int = fir.coordinate_of %h, alloc_int : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %poly = fir.coordinate_of %h, poly : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>
  %cp = fir.coordinate_of %h, cp : (!fir.ref<!fir.type<_QFstruct_membersTholder{alloc_arr:!fir.box<!fir.heap<!fir.array<?xf32>>>,ptr_arr:!fir.box<!fir.ptr<!fir.array<?xf32>>>,alloc_int:!fir.box<!fir.heap<i32>>,poly:!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>,cp:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  %c0 = acc.copyin varPtr(%alloc_arr : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
      name("h%alloc_arr") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %c1 = acc.copyin varPtr(%ptr_arr : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
      name("h%ptr_arr") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  %c2 = acc.copyin varPtr(%alloc_int : !fir.ref<!fir.box<!fir.heap<i32>>>)
      dataClause(acc_copy) name("h%alloc_int")
      -> !fir.ref<!fir.box<!fir.heap<i32>>>
  acc.copyout accPtr(%c2 : !fir.ref<!fir.box<!fir.heap<i32>>>)
      to varPtr(%alloc_int : !fir.ref<!fir.box<!fir.heap<i32>>>)
      dataClause(acc_copy) name("h%alloc_int")
  %c3 = acc.copyin varPtr(%poly : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>)
      name("h%poly") -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>
  %c4 = acc.copyin varPtr(%cp : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
      name("h%cp") -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
  acc.data dataOperands(%c0, %c1, %c2, %c3, %c4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QMm_baseTbase_t{tag:i32}>>>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
    acc.terminator
  }
  return
}

// Mapping a tuple of references transfers the tuple storage: every member is
// one target address, so this is two pointers (16).
// CHECK-LABEL: func.func @tuple_of_references
// CHECK: %[[SIZE:.*]] = arith.constant 16 : i64
// CHECK: acc.map_info varPtr(%arg0 : !fir.ref<tuple<!fir.ref<i32>, !fir.ref<f64>>>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: mapFlags(to,implicit)
func.func @tuple_of_references(
    %arg0: !fir.ref<tuple<!fir.ref<i32>, !fir.ref<f64>>>) {
  %copy = acc.copyin
      varPtr(%arg0 : !fir.ref<tuple<!fir.ref<i32>, !fir.ref<f64>>>)
      implicit(true) name("") -> !fir.ref<tuple<!fir.ref<i32>, !fir.ref<f64>>>
  acc.kernel_environment
      dataOperands(%copy : !fir.ref<tuple<!fir.ref<i32>, !fir.ref<f64>>>) {
  }
  return
}

// A derived type with no data components, such as one that only declares a
// type-bound procedure, has zero-sized storage.
// CHECK-LABEL: func.func @empty_record
// CHECK: %[[SIZE:.*]] = arith.constant 0 : i64
// CHECK: acc.map_info varPtr(%arg0 : !fir.ref<!fir.type<_QMm_emptyTempty_t>>)
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(0)
func.func @empty_record(%arg0: !fir.ref<!fir.type<_QMm_emptyTempty_t>>) {
  %copy = acc.copyin varPtr(%arg0 : !fir.ref<!fir.type<_QMm_emptyTempty_t>>)
      name("tt") -> !fir.ref<!fir.type<_QMm_emptyTempty_t>>
  acc.data dataOperands(%copy : !fir.ref<!fir.type<_QMm_emptyTempty_t>>) {
    acc.terminator
  }
  return
}
