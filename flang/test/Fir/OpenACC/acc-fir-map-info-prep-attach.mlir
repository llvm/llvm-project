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