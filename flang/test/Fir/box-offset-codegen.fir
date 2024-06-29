// Test fir.box_offset code generation.
// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

func.func @scalar_addr(%scalar : !fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<t>>> {
  %addr = fir.box_offset %scalar base_addr : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<t>>>
  return %addr : !fir.llvm_ptr<!fir.ref<!fir.type<t>>>
}
// CHECK-LABEL: define ptr @scalar_addr(
// CHECK-SAME: ptr %[[BOX:.*]]){{.*}}{
// CHECK:    %[[VAL_0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[BOX]], i32 0, i32 0
// CHECK:    ret ptr %[[VAL_0]]

func.func @scalar_tdesc(%scalar : !fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>> {
  %tdesc = fir.box_offset %scalar derived_type : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
  return %tdesc : !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
}
// CHECK-LABEL: define ptr @scalar_tdesc(
// CHECK-SAME: ptr %[[BOX:.*]]){{.*}}{
// CHECK:    %[[VAL_0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, ptr, [1 x i64] }, ptr %[[BOX]], i32 0, i32 7
// CHECK:    ret ptr %[[VAL_0]]

func.func @array_addr(%array : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.ptr<!fir.array<?x!fir.type<t>>>> {
  %addr = fir.box_offset %array base_addr : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.ptr<!fir.array<?x!fir.type<t>>>>
  return %addr : !fir.llvm_ptr<!fir.ptr<!fir.array<?x!fir.type<t>>>>
}
// CHECK-LABEL: define ptr @array_addr(
// CHECK-SAME: ptr %[[BOX:.*]]){{.*}}{
// CHECK:    %[[VAL_0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[BOX]], i32 0, i32 0
// CHECK:    ret ptr %[[VAL_0]]

func.func @array_tdesc(%array : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>> {
  %tdesc = fir.box_offset %array derived_type : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<t>>>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
  return %tdesc : !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
}
// CHECK-LABEL: define ptr @array_tdesc(
// CHECK-SAME: ptr %[[BOX:.*]]){{.*}}{
// CHECK:    %[[VAL_0:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[BOX]], i32 0, i32 8
// CHECK:    ret ptr %[[VAL_0]]
