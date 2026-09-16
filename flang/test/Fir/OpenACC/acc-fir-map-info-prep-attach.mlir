// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep))" | FileCheck %s
// RUN: fir-opt %s --pass-pipeline="builtin.module(func.func(acc-fir-map-info-prep,acc-fir-map-info-prep))" | FileCheck %s --check-prefix=IDEMP

// Attach metadata for descriptor-backed components. Distinguish the mapped
// pointee, optional separate desc value, and optional varPtrPtr attach slot.

// A component nested through more than one record still attaches through the
// immediate descriptor slot. Its descriptor and mapped pointee are distinct.
// CHECK-LABEL: func.func @nested_component
// CHECK: %[[INNER:.*]] = fir.coordinate_of %{{.*}}, inner
// CHECK: %[[SLOT:.*]] = fir.coordinate_of %[[INNER]], values
// CHECK: %[[BOX:.*]] = fir.load %[[SLOT]]
// CHECK: %[[DATA:.*]] = fir.box_addr %[[BOX]]
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.heap<!fir.array<?xf64>>)
// CHECK-SAME: varPtrPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>)
// CHECK-SAME: desc(%[[BOX]] : !fir.box<!fir.heap<!fir.array<?xf64>>>)
// CHECK-SAME: elementSize(8)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// IDEMP-LABEL: func.func @nested_component
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @nested_component() {
  %outer = fir.undefined !fir.ref<!fir.type<_QFattachTouter{inner:!fir.type<_QFattachTinner{values:!fir.box<!fir.heap<!fir.array<?xf64>>>}>}>>
  %inner = fir.coordinate_of %outer, inner : (!fir.ref<!fir.type<_QFattachTouter{inner:!fir.type<_QFattachTinner{values:!fir.box<!fir.heap<!fir.array<?xf64>>>}>}>>) -> !fir.ref<!fir.type<_QFattachTinner{values:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>
  %slot = fir.coordinate_of %inner, values : (!fir.ref<!fir.type<_QFattachTinner{values:!fir.box<!fir.heap<!fir.array<?xf64>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %box = fir.load %slot : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf64>>>>
  %data = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf64>>>) -> !fir.heap<!fir.array<?xf64>>
  %copy = acc.copyin varPtr(%data : !fir.heap<!fir.array<?xf64>>)
      dataClause(acc_copyin) name("outer%inner%values")
      -> !fir.heap<!fir.array<?xf64>>
  acc.data dataOperands(%copy : !fir.heap<!fir.array<?xf64>>) {
    acc.terminator
  }
  return
}

// Preserve an explicit varPtrPtr on an implicit entry: inference must not
// replace metadata already carried on the data entry.
// CHECK-LABEL: func.func @existing_attach_point
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.ptr<f32>>>
// CHECK: %[[BOX:.*]] = fir.load %[[SLOT]]
// CHECK: %[[DATA:.*]] = fir.box_addr %[[BOX]]
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.ptr<f32>)
// CHECK-SAME: varPtrPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.ptr<f32>>>)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to,ptr_and_obj,implicit)
// IDEMP-LABEL: func.func @existing_attach_point
func.func @existing_attach_point() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.ptr<f32>>>
  %box = fir.load %slot : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %data = fir.box_addr %box : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %copy = acc.copyin varPtr(%data : !fir.ptr<f32>)
      varPtrPtr(%slot : !fir.ref<!fir.box<!fir.ptr<f32>>>)
      dataClause(acc_copyin) implicit(true) name("p")
      -> !fir.ptr<f32>
  acc.data dataOperands(%copy : !fir.ptr<f32>) {
    acc.terminator
  }
  return
}

// A loaded descriptor of an entity that is neither POINTER nor ALLOCATABLE is
// not named: the specification leaves such a descriptor unmanaged on the
// device, so it does not describe the mapped object there. The entry is a plain
// object map, sized from its type.
// CHECK-LABEL: func.func @loaded_unmanaged_descriptor
// CHECK: %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.array<100xf32>>>
// CHECK: %[[DATA:.*]] = fir.box_addr %[[BOX]]
// CHECK: %[[SIZE:.*]] = arith.constant 400 : i64
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.ref<!fir.array<100xf32>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// IDEMP-LABEL: func.func @loaded_unmanaged_descriptor
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @loaded_unmanaged_descriptor() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.array<100xf32>>>
  %box = fir.load %slot : !fir.ref<!fir.box<!fir.array<100xf32>>>
  %data = fir.box_addr %box : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
  %copy = acc.copyin varPtr(%data : !fir.ref<!fir.array<100xf32>>)
      dataClause(acc_copyin) name("v") -> !fir.ref<!fir.array<100xf32>>
  acc.data dataOperands(%copy : !fir.ref<!fir.array<100xf32>>) {
    acc.terminator
  }
  return
}

// An assumed-shape dummy arrives as a box parameter. Mapping that box is a
// descriptor map recovered from var: no separate desc operand and no attach
// slot, because the specification leaves this descriptor unmanaged on the
// device.
// CHECK-LABEL: func.func @assumed_shape_dummy
// CHECK-SAME: %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>
// CHECK: acc.map_info var(%[[BOX]] : !fir.box<!fir.array<?xf32>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: ptr_and_obj
// IDEMP-LABEL: func.func @assumed_shape_dummy
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @assumed_shape_dummy(%box: !fir.box<!fir.array<?xf32>>) {
  %copy = acc.copyin var(%box : !fir.box<!fir.array<?xf32>>)
      dataClause(acc_copyin) name("a") -> !fir.box<!fir.array<?xf32>>
  acc.data dataOperands(%copy : !fir.box<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// The base address taken out of that same box parameter is a plain object map:
// there is no descriptor slot to attach through, and the box itself is not
// named. An explicit clause keeps the unknown size of a dynamic extent.
// CHECK-LABEL: func.func @assumed_shape_base_address
// CHECK-SAME: %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>
// CHECK: %[[DATA:.*]] = fir.box_addr %[[BOX]]
// CHECK: %[[SIZE:.*]] = arith.constant -1 : i64
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.ref<!fir.array<?xf32>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// IDEMP-LABEL: func.func @assumed_shape_base_address
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @assumed_shape_base_address(%box: !fir.box<!fir.array<?xf32>>) {
  %data = fir.box_addr %box : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %copy = acc.copyin varPtr(%data : !fir.ref<!fir.array<?xf32>>)
      dataClause(acc_copyin) name("a") -> !fir.ref<!fir.array<?xf32>>
  acc.data dataOperands(%copy : !fir.ref<!fir.array<?xf32>>) {
    acc.terminator
  }
  return
}

// A polymorphic dummy arrives as a class parameter. It behaves like the
// assumed-shape box: mapping the class is a descriptor map recovered from var,
// with no separate desc operand and no attach slot.
// CHECK-LABEL: func.func @polymorphic_dummy
// CHECK-SAME: %[[CLASS:.*]]: !fir.class<!fir.type<_QMtypesTbase{i:i32}>>
// CHECK: acc.map_info var(%[[CLASS]] : !fir.class<!fir.type<_QMtypesTbase{i:i32}>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to)
// CHECK-NOT: ptr_and_obj
// IDEMP-LABEL: func.func @polymorphic_dummy
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @polymorphic_dummy(%class: !fir.class<!fir.type<_QMtypesTbase{i:i32}>>) {
  %copy = acc.copyin var(%class : !fir.class<!fir.type<_QMtypesTbase{i:i32}>>)
      dataClause(acc_copyin) name("p") -> !fir.class<!fir.type<_QMtypesTbase{i:i32}>>
  acc.data dataOperands(%copy : !fir.class<!fir.type<_QMtypesTbase{i:i32}>>) {
    acc.terminator
  }
  return
}

// The base address taken out of that class parameter is likewise a plain
// object map, sized from the record type.
// CHECK-LABEL: func.func @polymorphic_base_address
// CHECK-SAME: %[[CLASS:.*]]: !fir.class<!fir.type<_QMtypesTbase{i:i32}>>
// CHECK: %[[DATA:.*]] = fir.box_addr %[[CLASS]]
// CHECK: %[[SIZE:.*]] = arith.constant 4 : i64
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.ref<!fir.type<_QMtypesTbase{i:i32}>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: size(%[[SIZE]] : i64)
// CHECK-SAME: descKind(none)
// CHECK-SAME: mapFlags(to)
// IDEMP-LABEL: func.func @polymorphic_base_address
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @polymorphic_base_address(%class: !fir.class<!fir.type<_QMtypesTbase{i:i32}>>) {
  %data = fir.box_addr %class : (!fir.class<!fir.type<_QMtypesTbase{i:i32}>>) -> !fir.ref<!fir.type<_QMtypesTbase{i:i32}>>
  %copy = acc.copyin varPtr(%data : !fir.ref<!fir.type<_QMtypesTbase{i:i32}>>)
      dataClause(acc_copyin) name("p") -> !fir.ref<!fir.type<_QMtypesTbase{i:i32}>>
  acc.data dataOperands(%copy : !fir.ref<!fir.type<_QMtypesTbase{i:i32}>>) {
    acc.terminator
  }
  return
}

// A polymorphic ALLOCATABLE is a class whose descriptor the specification does
// require to be maintained on the device, so the pointee map keeps its attach
// slot and names the descriptor. The distinction is the entity, not whether
// the descriptor is a box or a class.
// CHECK-LABEL: func.func @polymorphic_allocatable
// CHECK-SAME: %[[SLOT:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>>
// CHECK: %[[CLASS:.*]] = fir.load %[[SLOT]]
// CHECK: %[[DATA:.*]] = fir.box_addr %[[CLASS]]
// CHECK: acc.map_info varPtr(%[[DATA]] : !fir.heap<!fir.type<_QMtypesTbase{i:i32}>>)
// CHECK-SAME: varPtrPtr(%[[SLOT]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>>)
// CHECK-SAME: desc(%[[CLASS]] : !fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
// IDEMP-LABEL: func.func @polymorphic_allocatable
// IDEMP-COUNT-1: acc.map_info
// IDEMP-NOT: acc.copyin
func.func @polymorphic_allocatable(%slot: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>>) {
  %box = fir.load %slot : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>>
  %data = fir.box_addr %box : (!fir.class<!fir.heap<!fir.type<_QMtypesTbase{i:i32}>>>) -> !fir.heap<!fir.type<_QMtypesTbase{i:i32}>>
  %copy = acc.copyin varPtr(%data : !fir.heap<!fir.type<_QMtypesTbase{i:i32}>>)
      dataClause(acc_copyin) name("p") -> !fir.heap<!fir.type<_QMtypesTbase{i:i32}>>
  acc.data dataOperands(%copy : !fir.heap<!fir.type<_QMtypesTbase{i:i32}>>) {
    acc.terminator
  }
  return
}

// Mapping descriptor storage itself has no second indirection operand. The
// descriptor is recovered from var and supplies both CFI and ptr_and_obj facts.
// CHECK-LABEL: func.func @descriptor_storage
// CHECK: %[[SLOT:.*]] = fir.undefined !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK: acc.map_info varPtr(%[[SLOT]] : !fir.ref<!fir.box<!fir.ptr<i32>>>)
// CHECK-NOT: varPtrPtr
// CHECK-NOT: desc(
// CHECK-SAME: elementSize(4)
// CHECK-SAME: descKind(cfi)
// CHECK-SAME: mapFlags(to,ptr_and_obj)
func.func @descriptor_storage() {
  %slot = fir.undefined !fir.ref<!fir.box<!fir.ptr<i32>>>
  %copy = acc.copyin varPtr(%slot : !fir.ref<!fir.box<!fir.ptr<i32>>>)
      dataClause(acc_copyin) name("p") -> !fir.ref<!fir.box<!fir.ptr<i32>>>
  acc.data dataOperands(%copy : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
    acc.terminator
  }
  return
}