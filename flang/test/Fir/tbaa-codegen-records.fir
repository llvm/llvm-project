// RUN: fir-opt --split-input-file --pass-pipeline="builtin.module(fir-to-llvm-ir{apply-tbaa=true})" %s | FileCheck %s

// Ensure that records with array members are identified as having descriptor members,
// as reflected by the TBAA metadata.

func.func @record_array_member(%arg0 : !fir.ref<!fir.type<_QFTt2{y:!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>}>>) {
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %1 = fir.alloca !fir.type<_QFTt2{y:!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>}> {bindc_name = "z", uniq_name = "_QFB1Ez"}
  fir.copy %arg0 to %1 no_overlap : !fir.ref<!fir.type<_QFTt2{y:!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>}>>, !fir.ref<!fir.type<_QFTt2{y:!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>}>>
  %3 = fir.coordinate_of %1, y : (!fir.ref<!fir.type<_QFTt2{y:!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>}>>) -> !fir.ref<!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
  %4 = fircg.ext_array_coor %3(%c1)<%c1> : (!fir.ref<!fir.array<1x!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, index, index) -> !fir.ref<!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  %5 = fir.coordinate_of %4, x : (!fir.ref<!fir.type<_QFTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %6 = fir.load %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}

// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root record_array_member">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "descriptor member", members = {<#tbaa_type_desc, 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>

// CHECK-LABEL:   llvm.func @record_array_member(
// CHECK-SAME:                                   %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
// CHECK:           %[[X_VAL:.*]] = llvm.alloca %{{.+}} x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[Z_VAL:.*]] = llvm.alloca %{{.+}} x !llvm.struct<"_QFTt2", (array<1 x struct<"_QFTt", (struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>)>>)> {bindc_name = "z"} : (i64) -> !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[Z_VAL]], %[[ARG0]], %{{.+}}) <{isVolatile = false, tbaa = [#[[$ATTR_2]]]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()

// CHECK:           "llvm.intr.memcpy"(%[[X_VAL]], %{{.+}}, %{{.+}}) <{isVolatile = false, tbaa = [#[[$ATTR_4]]]}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
