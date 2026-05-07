// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = #llvm.access_group<id = distinct[0]<>>
// CHECK: #[[$ATTR_1:.+]] = #llvm.loop_annotation<parallelAccesses = #[[$ATTR_0]]>
// CHECK-LABEL:   llvm.func @test_(
// CHECK-SAME:                     %[[ARG0:.*]]: !llvm.ptr {fir.bindc_name = "n", llvm.noalias, llvm.nocapture},
// CHECK-SAME:                     %[[ARG1:.*]]: !llvm.ptr {fir.bindc_name = "m", llvm.noalias, llvm.nocapture},
// CHECK-SAME:                     %[[ARG2:.*]]: !llvm.ptr {fir.bindc_name = "k", llvm.noalias, llvm.nocapture},
// CHECK-SAME:                     %[[ARG3:.*]]: !llvm.ptr {fir.bindc_name = "v", fir.contiguous, llvm.nocapture}) attributes {fir.internal_name = "_QPtest", target_cpu = "x86-64"} {
// CHECK:           %[[MLIR_0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[ALLOCA_0:.*]] = llvm.alloca %[[MLIR_0]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[MLIR_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[ALLOCA_1:.*]] = llvm.alloca %[[MLIR_1]] x i32 {bindc_name = "j"} : (i64) -> !llvm.ptr
// CHECK:           %[[MLIR_2:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[MLIR_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[MLIR_4:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[LOAD_0:.*]] = llvm.load %[[ARG2]] : !llvm.ptr -> i32
// CHECK:           %[[SEXT_0:.*]] = llvm.sext %[[LOAD_0]] : i32 to i64
// CHECK:           %[[TRUNC_0:.*]] = llvm.trunc %[[MLIR_3]] : i64 to i32
// CHECK:           llvm.br ^bb1(%[[TRUNC_0]], %[[SEXT_0]] : i32, i64)
// CHECK:         ^bb1(%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i64):
// CHECK:           %[[ICMP_0:.*]] = llvm.icmp "sgt" %[[VAL_1]], %[[MLIR_2]] : i64
// CHECK:           llvm.cond_br %[[ICMP_0]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           llvm.store %[[VAL_0]], %[[ALLOCA_1]] {access_groups = [#[[$ATTR_0]]]} : i32, !llvm.ptr
// CHECK:           %[[LOAD_1:.*]] = llvm.load %[[ALLOCA_1]] {accessGroups = [#[[$ATTR_0]]], access_groups = [#[[$ATTR_0]]]} : !llvm.ptr -> i32
// CHECK:           %[[LOAD_2:.*]] = llvm.load %[[ARG1]] {accessGroups = [#[[$ATTR_0]]], access_groups = [#[[$ATTR_0]]]} : !llvm.ptr -> i32
// CHECK:           %[[ADD_0:.*]] = llvm.add %[[LOAD_1]], %[[LOAD_2]] : i32
// CHECK:           %[[SITOFP_0:.*]] = llvm.sitofp %[[ADD_0]] : i32 to f32
// CHECK:           %[[MLIR_5:.*]] = llvm.mlir.constant(72 : i32) : i32
// CHECK:           "llvm.intr.memcpy"(%[[ALLOCA_0]], %[[ARG3]], %[[MLIR_5]]) <{access_groups = [#[[$ATTR_0]]], arg_attrs = [{llvm.align = 8 : i64}, {llvm.align = 8 : i64}, {}], isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
// CHECK:           %[[LOAD_3:.*]] = llvm.load %[[ARG0]] {accessGroups = [#[[$ATTR_0]]], access_groups = [#[[$ATTR_0]]]} : !llvm.ptr -> i32
// CHECK:           %[[SEXT_1:.*]] = llvm.sext %[[LOAD_3]] : i32 to i64
// CHECK:           %[[SEXT_2:.*]] = llvm.sext %[[LOAD_1]] : i32 to i64
// CHECK:           %[[GETELEMENTPTR_0:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_4:.*]] = llvm.load %[[GETELEMENTPTR_0]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[GETELEMENTPTR_1:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_2]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_5:.*]] = llvm.load %[[GETELEMENTPTR_1]] : !llvm.ptr -> i64
// CHECK:           %[[GETELEMENTPTR_2:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_2]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_6:.*]] = llvm.load %[[GETELEMENTPTR_2]] : !llvm.ptr -> i64
// CHECK:           %[[GETELEMENTPTR_3:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_2]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_7:.*]] = llvm.load %[[GETELEMENTPTR_3]] : !llvm.ptr -> i64
// CHECK:           %[[GETELEMENTPTR_4:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_3]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_8:.*]] = llvm.load %[[GETELEMENTPTR_4]] : !llvm.ptr -> i64
// CHECK:           %[[GETELEMENTPTR_5:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_3]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_9:.*]] = llvm.load %[[GETELEMENTPTR_5]] : !llvm.ptr -> i64
// CHECK:           %[[GETELEMENTPTR_6:.*]] = llvm.getelementptr %[[ALLOCA_0]][0, 7, %[[MLIR_3]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[LOAD_10:.*]] = llvm.load %[[GETELEMENTPTR_6]] : !llvm.ptr -> i64
// CHECK:           %[[MLIR_6:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[MLIR_7:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[SUB_0:.*]] = llvm.sub %[[SEXT_1]], %[[LOAD_5]] overflow<nsw> : i64
// CHECK:           %[[MUL_0:.*]] = llvm.mul %[[SUB_0]], %[[MLIR_6]] overflow<nsw, nuw> : i64
// CHECK:           %[[MUL_1:.*]] = llvm.mul %[[MUL_0]], %[[MLIR_6]] overflow<nsw, nuw> : i64
// CHECK:           %[[ADD_1:.*]] = llvm.add %[[MUL_1]], %[[MLIR_7]] overflow<nsw, nuw> : i64
// CHECK:           %[[MUL_2:.*]] = llvm.mul %[[MLIR_6]], %[[LOAD_6]] overflow<nsw, nuw> : i64
// CHECK:           %[[SUB_1:.*]] = llvm.sub %[[SEXT_2]], %[[LOAD_8]] overflow<nsw> : i64
// CHECK:           %[[MUL_3:.*]] = llvm.mul %[[SUB_1]], %[[MLIR_6]] overflow<nsw, nuw> : i64
// CHECK:           %[[MUL_4:.*]] = llvm.mul %[[MUL_3]], %[[MUL_2]] overflow<nsw, nuw> : i64
// CHECK:           %[[ADD_2:.*]] = llvm.add %[[MUL_4]], %[[ADD_1]] overflow<nsw, nuw> : i64
// CHECK:           %[[MUL_5:.*]] = llvm.mul %[[MUL_2]], %[[LOAD_9]] overflow<nsw, nuw> : i64
// CHECK:           %[[GETELEMENTPTR_7:.*]] = llvm.getelementptr nusw|nuw %[[LOAD_4]]{{\[}}%[[ADD_2]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:           llvm.store %[[SITOFP_0]], %[[GETELEMENTPTR_7]] {access_groups = [#[[$ATTR_0]]]} : f32, !llvm.ptr
// CHECK:           %[[LOAD_11:.*]] = llvm.load %[[ALLOCA_1]] {accessGroups = [#[[$ATTR_0]]], access_groups = [#[[$ATTR_0]]]} : !llvm.ptr -> i32
// CHECK:           %[[ADD_3:.*]] = llvm.add %[[LOAD_11]], %[[TRUNC_0]] overflow<nsw> : i32
// CHECK:           %[[SUB_2:.*]] = llvm.sub %[[VAL_1]], %[[MLIR_3]] : i64
// CHECK:           llvm.br ^bb1(%[[ADD_3]], %[[SUB_2]] : i32, i64) {loop_annotation = #[[$ATTR_1]]}
// CHECK:         ^bb3:
// CHECK:           llvm.store %[[VAL_0]], %[[ALLOCA_1]] : i32, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }

#access_group = #llvm.access_group<id = distinct[0]<>>
#loop_annotation = #llvm.loop_annotation<parallelAccesses = #access_group>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "x86-64", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (https://github.com/llvm/llvm-project.git b97d4a685a02023941d4b17e5c0e849f8e313e06)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @test_(%arg0: !fir.ref<i32> {fir.bindc_name = "n", llvm.noalias, llvm.nocapture}, %arg1: !fir.ref<i32> {fir.bindc_name = "m", llvm.noalias, llvm.nocapture}, %arg2: !fir.ref<i32> {fir.bindc_name = "k", llvm.noalias, llvm.nocapture}, %arg3: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "v", fir.contiguous, llvm.nocapture}) attributes {fir.internal_name = "_QPtest", llvm.target_cpu = "x86-64"} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtestEj"}
    %1 = fir.load %arg2 : !fir.ref<i32>
    %2 = fir.convert %1 : (i32) -> index
    %3 = fir.convert %c1 : (index) -> i32
    cf.br ^bb1(%3, %2 : i32, index)
  ^bb1(%4: i32, %5: index):  // 2 preds: ^bb0, ^bb2
    %6 = arith.cmpi sgt, %5, %c0 : index
    cf.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    fir.store %4 to %0 {accessGroups = [#access_group]} : !fir.ref<i32>
    %7 = fir.load %0 {accessGroups = [#access_group]} : !fir.ref<i32>
    %8 = fir.load %arg1 {accessGroups = [#access_group]} : !fir.ref<i32>
    %9 = arith.addi %7, %8 : i32
    %10 = fir.convert %9 : (i32) -> f32
    %11 = fir.load %arg3 {accessGroups = [#access_group]} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
    %12 = fir.load %arg0 {accessGroups = [#access_group]} : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.convert %7 : (i32) -> i64
    %15 = fir.box_addr %11 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>) -> !fir.ptr<!fir.array<?x?xf32>>
    %16:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
    %17:3 = fir.box_dims %11, %c1 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
    %18 = fircg.ext_array_coor %15(%16#1, %17#1) origin %16#0, %17#0<%13, %14> : (!fir.ptr<!fir.array<?x?xf32>>, index, index, index, index, i64, i64) -> !fir.ref<f32>
    fir.store %10 to %18 {accessGroups = [#access_group]} : !fir.ref<f32>
    %19 = fir.load %0 {accessGroups = [#access_group]} : !fir.ref<i32>
    %20 = arith.addi %19, %3 overflow<nsw> : i32
    %21 = arith.subi %5, %c1 : index
    cf.br ^bb1(%20, %21 : i32, index) {loop_annotation = #loop_annotation}
  ^bb3:  // pred: ^bb1
    fir.store %4 to %0 : !fir.ref<i32>
    return
  }
}
