// Test fir.box_offset parse/print/parse/print identity.
// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @test_box_offset(%unlimited : !fir.ref<!fir.class<none>>, %type_star : !fir.ref<!fir.box<!fir.array<?xnone>>>) {
  %box1 = fir.alloca !fir.box<i32>
  %addr1 = fir.box_offset %box1 base_addr : (!fir.ref<!fir.box<i32>>) -> !fir.llvm_ptr<!fir.ref<i32>>

  %box2 = fir.alloca !fir.box<!fir.type<t>>
  %addr2 = fir.box_offset %box2 base_addr : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<t>>>
  %tdesc2 = fir.box_offset %box2 derived_type : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>

  %box3 = fir.alloca !fir.box<!fir.array<?xi32>>
  %addr3 = fir.box_offset %box3 base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>

  %box4 = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>
  %addr4 = fir.box_offset %box4 base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.ptr<!fir.array<?x!fir.type<t>>>>
  %tdesc4 = fir.box_offset %box4 derived_type : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>

  %addr5 = fir.box_offset %unlimited base_addr : (!fir.ref<!fir.class<none>>) -> !fir.llvm_ptr<!fir.ref<none>>
  %tdesc5 = fir.box_offset %unlimited derived_type : (!fir.ref<!fir.class<none>>) -> !fir.llvm_ptr<!fir.tdesc<none>>

  %addr6 = fir.box_offset %type_star base_addr : (!fir.ref<!fir.box<!fir.array<?xnone>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xnone>>>
  %tdesc6 = fir.box_offset %type_star derived_type : (!fir.ref<!fir.box<!fir.array<?xnone>>>) -> !fir.llvm_ptr<!fir.tdesc<none>>
  return
}
// CHECK-LABEL:   func.func @test_box_offset(
// CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.class<none>>,
// CHECK-SAME:                               %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.array<?xnone>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<i32>
// CHECK:           %[[VAL_3:.*]] = fir.box_offset %[[VAL_2]] base_addr : (!fir.ref<!fir.box<i32>>) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.type<t>>
// CHECK:           %[[VAL_5:.*]] = fir.box_offset %[[VAL_4]] base_addr : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<t>>>
// CHECK:           %[[VAL_6:.*]] = fir.box_offset %[[VAL_4]] derived_type : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_8:.*]] = fir.box_offset %[[VAL_7]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:           %[[VAL_9:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>
// CHECK:           %[[VAL_10:.*]] = fir.box_offset %[[VAL_9]] base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.ptr<!fir.array<?x!fir.type<t>>>>
// CHECK:           %[[VAL_11:.*]] = fir.box_offset %[[VAL_9]] derived_type : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
// CHECK:           %[[VAL_12:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.class<none>>) -> !fir.llvm_ptr<!fir.ref<none>>
// CHECK:           %[[VAL_13:.*]] = fir.box_offset %[[VAL_0]] derived_type : (!fir.ref<!fir.class<none>>) -> !fir.llvm_ptr<!fir.tdesc<none>>
// CHECK:           %[[VAL_14:.*]] = fir.box_offset %[[VAL_1]] base_addr : (!fir.ref<!fir.box<!fir.array<?xnone>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xnone>>>
// CHECK:           %[[VAL_15:.*]] = fir.box_offset %[[VAL_1]] derived_type : (!fir.ref<!fir.box<!fir.array<?xnone>>>) -> !fir.llvm_ptr<!fir.tdesc<none>>
