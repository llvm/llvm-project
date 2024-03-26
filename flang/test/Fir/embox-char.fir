// Test that the offset of the first element of the slice
// is computed in elements of the type used for the GEP
// computing the base of the slice.

// RUN: fir-opt -o - -cg-rewrite --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco -o - -cg-rewrite --fir-to-llvm-ir %s | FileCheck %s

// subroutine test(x)
//   character(:,ck),allocatable :: x(:,:)
//   print *, x(2,:)
// end subroutine

// CHECK-LABEL:   llvm.func @test_char4(
// CHECK-SAME:        %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:        %[[VAL_1_SLICE_LB0:.*]]: i64, %[[VAL_2_SLICE_EX0:.*]]: i64, %[[VAL_3_SLICE_ST0:.*]]: i64, %[[VAL_4_SLICE_LB1:.*]]: i64, %[[VAL_5_SLICE_EX1:.*]]: i64, %[[VAL_6_SLICE_ST1:.*]]: i64) {
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_9]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_13_WIDTH:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK:           %[[VAL_14:.*]] = ptr.load %[[VAL_0]] : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           ptr.store %[[VAL_14]], %[[VAL_10]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>, !llvm.ptr
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_10]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_16_BYTESIZE:.*]] = ptr.load %[[VAL_15]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_18_LB1:.*]] = ptr.load %[[VAL_17]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_20_EX1:.*]] = ptr.load %[[VAL_19]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_22_ST1:.*]] = ptr.load %[[VAL_21]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_10]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_24_BASEPTR:.*]] = ptr.load %[[VAL_23]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_26_LB0:.*]] = ptr.load %[[VAL_25]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_28_EX0:.*]] = ptr.load %[[VAL_27]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_30_ST0:.*]] = ptr.load %[[VAL_29]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_31_LEN:.*]] = llvm.sdiv %[[VAL_16_BYTESIZE]], %[[VAL_13_WIDTH]]  : i64
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(44 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_34:.*]] = llvm.getelementptr %[[VAL_33]][1] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK:           %[[VAL_35:.*]] = ptr.ptrtoint %[[VAL_34]] : !llvm.ptr to i64
// CHECK:           %[[VAL_36_BYTESIZE:.*]] = llvm.mul %[[VAL_35]], %[[VAL_31_LEN]]  : i64
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_36_BYTESIZE]], %[[VAL_37]][1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:           %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_39]], %[[VAL_38]][2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_42:.*]] = llvm.trunc %[[VAL_41]] : i32 to i8
// CHECK:           %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_42]], %[[VAL_40]][3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_44:.*]] = llvm.trunc %[[VAL_32]] : i32 to i8
// CHECK:           %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_43]][4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.trunc %[[VAL_46]] : i32 to i8
// CHECK:           %[[VAL_48:.*]] = llvm.insertvalue %[[VAL_47]], %[[VAL_45]][5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_50:.*]] = llvm.trunc %[[VAL_49]] : i32 to i8
// CHECK:           %[[VAL_51:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_48]][6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_52_c0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_53:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_54:.*]] = llvm.sub %[[VAL_1_SLICE_LB0]], %[[VAL_26_LB0]]  : i64
// CHECK:           %[[VAL_55:.*]] = llvm.mul %[[VAL_54]], %[[VAL_31_LEN]]  : i64
// CHECK:           %[[VAL_56_SLICE_OFF0:.*]] = llvm.add %[[VAL_55]], %[[VAL_52_c0]]  : i64
// CHECK:           %[[VAL_57:.*]] = llvm.sub %[[VAL_2_SLICE_EX0]], %[[VAL_1_SLICE_LB0]]  : i64
// CHECK:           %[[VAL_58:.*]] = llvm.add %[[VAL_57]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_59:.*]] = llvm.sdiv %[[VAL_58]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_60:.*]] = llvm.icmp "sgt" %[[VAL_59]], %[[VAL_52_c0]] : i64
// CHECK:           %[[VAL_61:.*]] = llvm.select %[[VAL_60]], %[[VAL_59]], %[[VAL_52_c0]] : i1, i64
// CHECK:           %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_53]], %[[VAL_51]][7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_61]], %[[VAL_62]][7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_64:.*]] = llvm.mul %[[VAL_36_BYTESIZE]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_63]][7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_66:.*]] = llvm.mul %[[VAL_36_BYTESIZE]], %[[VAL_28_EX0]]  : i64
// CHECK:           %[[VAL_67:.*]] = llvm.mul %[[VAL_31_LEN]], %[[VAL_28_EX0]]  : i64
// CHECK:           %[[VAL_68:.*]] = llvm.sub %[[VAL_4_SLICE_LB1]], %[[VAL_18_LB1]]  : i64
// CHECK:           %[[VAL_69_SLICE_OFF1:.*]] = llvm.mul %[[VAL_68]], %[[VAL_67]]  : i64
// CHECK:           %[[VAL_70_OFFSET:.*]] = llvm.add %[[VAL_69_SLICE_OFF1]], %[[VAL_56_SLICE_OFF0]]  : i64
// CHECK:           %[[VAL_71:.*]] = llvm.sub %[[VAL_5_SLICE_EX1]], %[[VAL_4_SLICE_LB1]]  : i64
// CHECK:           %[[VAL_72:.*]] = llvm.add %[[VAL_71]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_73:.*]] = llvm.sdiv %[[VAL_72]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_74:.*]] = llvm.icmp "sgt" %[[VAL_73]], %[[VAL_52_c0]] : i64
// CHECK:           %[[VAL_75:.*]] = llvm.select %[[VAL_74]], %[[VAL_73]], %[[VAL_52_c0]] : i1, i64
// CHECK:           %[[VAL_76:.*]] = llvm.insertvalue %[[VAL_53]], %[[VAL_65]][7, 1, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_77:.*]] = llvm.insertvalue %[[VAL_75]], %[[VAL_76]][7, 1, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_78:.*]] = llvm.mul %[[VAL_66]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_79:.*]] = llvm.insertvalue %[[VAL_78]], %[[VAL_77]][7, 1, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_80:.*]] = llvm.mul %[[VAL_66]], %[[VAL_20_EX1]]  : i64
// CHECK:           %[[VAL_81:.*]] = llvm.mul %[[VAL_67]], %[[VAL_20_EX1]]  : i64
// CHECK:           %[[VAL_82:.*]] = llvm.getelementptr %[[VAL_24_BASEPTR]]{{\[}}%[[VAL_70_OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           %[[VAL_84:.*]] = llvm.insertvalue %[[VAL_82]], %[[VAL_79]][0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           ptr.store %[[VAL_84]], %[[VAL_8]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
func.func @test_char4(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %3 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>>
  %4 = fir.box_elesize %3 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>) -> index
  %5:3 = fir.box_dims %3, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>, index) -> (index, index, index)
  %8 = fir.box_addr %3 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>) -> !fir.heap<!fir.array<?x?x!fir.char<4,?>>>
  %9:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<4,?>>>>, index) -> (index, index, index)
  %10 = arith.divsi %4, %c4 : index
  %12 = fircg.ext_embox %8(%9#1, %5#1) origin %9#0, %5#0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6] typeparams %10 : (!fir.heap<!fir.array<?x?x!fir.char<4,?>>>, index, index, index, index, index, index, index, index, index, index, index) -> !fir.box<!fir.array<?x!fir.char<4,?>>>
  return
}

// CHECK-LABEL:   llvm.func @test_char1(
// CHECK-SAME:        %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:        %[[VAL_1_SLICE_LB0:.*]]: i64, %[[VAL_2_SLICE_EX0:.*]]: i64, %[[VAL_3_SLICE_ST0:.*]]: i64, %[[VAL_4_SLICE_LB1:.*]]: i64, %[[VAL_5_SLICE_EX1:.*]]: i64, %[[VAL_6_SLICE_ST1:.*]]: i64) {
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_7]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.alloca %[[VAL_9]] x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_12_c1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_14:.*]] = ptr.load %[[VAL_0]] : !llvm.ptr -> !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           ptr.store %[[VAL_14]], %[[VAL_10]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>, !llvm.ptr
// CHECK:           %[[VAL_15:.*]] = llvm.getelementptr %[[VAL_10]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_16_BYTESIZE:.*]] = ptr.load %[[VAL_15]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_18_LB1:.*]] = ptr.load %[[VAL_17]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_19:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_20_EX1:.*]] = ptr.load %[[VAL_19]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_12]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_22_ST1:.*]] = ptr.load %[[VAL_21]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_23:.*]] = llvm.getelementptr %[[VAL_10]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_24_BASEPTR:.*]] = ptr.load %[[VAL_23]] : !llvm.ptr -> !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_26_LB0:.*]] = ptr.load %[[VAL_25]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_28_EX0:.*]] = ptr.load %[[VAL_27]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_10]][0, 7, %[[VAL_11]], 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_30_ST0:.*]] = ptr.load %[[VAL_29]] : !llvm.ptr -> i64
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(40 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_34:.*]] = llvm.getelementptr %[[VAL_33]][1] : (!llvm.ptr) -> !llvm.ptr, i8
// CHECK:           %[[VAL_35:.*]] = ptr.ptrtoint %[[VAL_34]] : !llvm.ptr to i64
// CHECK:           %[[VAL_36_BYTESIZE:.*]] = llvm.mul %[[VAL_35]], %[[VAL_16_BYTESIZE]]  : i64
// CHECK:           %[[VAL_37:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_38:.*]] = llvm.insertvalue %[[VAL_36_BYTESIZE]], %[[VAL_37]][1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(20180515 : i32) : i32
// CHECK:           %[[VAL_40:.*]] = llvm.insertvalue %[[VAL_39]], %[[VAL_38]][2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_41:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_42:.*]] = llvm.trunc %[[VAL_41]] : i32 to i8
// CHECK:           %[[VAL_43:.*]] = llvm.insertvalue %[[VAL_42]], %[[VAL_40]][3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_44:.*]] = llvm.trunc %[[VAL_32]] : i32 to i8
// CHECK:           %[[VAL_45:.*]] = llvm.insertvalue %[[VAL_44]], %[[VAL_43]][4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.trunc %[[VAL_46]] : i32 to i8
// CHECK:           %[[VAL_48:.*]] = llvm.insertvalue %[[VAL_47]], %[[VAL_45]][5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_50:.*]] = llvm.trunc %[[VAL_49]] : i32 to i8
// CHECK:           %[[VAL_51:.*]] = llvm.insertvalue %[[VAL_50]], %[[VAL_48]][6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_52_c0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           %[[VAL_53:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           %[[VAL_54:.*]] = llvm.sub %[[VAL_1_SLICE_LB0]], %[[VAL_26_LB0]]  : i64
// CHECK:           %[[VAL_55:.*]] = llvm.mul %[[VAL_54]], %[[VAL_16_BYTESIZE]]  : i64
// CHECK:           %[[VAL_56_SLICE_OFF0:.*]] = llvm.add %[[VAL_55]], %[[VAL_52_c0]]  : i64
// CHECK:           %[[VAL_57:.*]] = llvm.sub %[[VAL_2_SLICE_EX0]], %[[VAL_1_SLICE_LB0]]  : i64
// CHECK:           %[[VAL_58:.*]] = llvm.add %[[VAL_57]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_59:.*]] = llvm.sdiv %[[VAL_58]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_60:.*]] = llvm.icmp "sgt" %[[VAL_59]], %[[VAL_52_c0]] : i64
// CHECK:           %[[VAL_61:.*]] = llvm.select %[[VAL_60]], %[[VAL_59]], %[[VAL_52_c0]] : i1, i64
// CHECK:           %[[VAL_62:.*]] = llvm.insertvalue %[[VAL_53]], %[[VAL_51]][7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_63:.*]] = llvm.insertvalue %[[VAL_61]], %[[VAL_62]][7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_64:.*]] = llvm.mul %[[VAL_36_BYTESIZE]], %[[VAL_3_SLICE_ST0]]  : i64
// CHECK:           %[[VAL_65:.*]] = llvm.insertvalue %[[VAL_64]], %[[VAL_63]][7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_66:.*]] = llvm.mul %[[VAL_36_BYTESIZE]], %[[VAL_28_EX0]]  : i64
// CHECK:           %[[VAL_67:.*]] = llvm.mul %[[VAL_16_BYTESIZE]], %[[VAL_28_EX0]]  : i64
// CHECK:           %[[VAL_68:.*]] = llvm.sub %[[VAL_4_SLICE_LB1]], %[[VAL_18_LB1]]  : i64
// CHECK:           %[[VAL_69_SLICE_OFF1:.*]] = llvm.mul %[[VAL_68]], %[[VAL_67]]  : i64
// CHECK:           %[[VAL_70_OFFSET:.*]] = llvm.add %[[VAL_69_SLICE_OFF1]], %[[VAL_56_SLICE_OFF0]]  : i64
// CHECK:           %[[VAL_71:.*]] = llvm.sub %[[VAL_5_SLICE_EX1]], %[[VAL_4_SLICE_LB1]]  : i64
// CHECK:           %[[VAL_72:.*]] = llvm.add %[[VAL_71]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_73:.*]] = llvm.sdiv %[[VAL_72]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_74:.*]] = llvm.icmp "sgt" %[[VAL_73]], %[[VAL_52_c0]] : i64
// CHECK:           %[[VAL_75:.*]] = llvm.select %[[VAL_74]], %[[VAL_73]], %[[VAL_52_c0]] : i1, i64
// CHECK:           %[[VAL_76:.*]] = llvm.insertvalue %[[VAL_53]], %[[VAL_65]][7, 1, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_77:.*]] = llvm.insertvalue %[[VAL_75]], %[[VAL_76]][7, 1, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_78:.*]] = llvm.mul %[[VAL_66]], %[[VAL_6_SLICE_ST1]]  : i64
// CHECK:           %[[VAL_79:.*]] = llvm.insertvalue %[[VAL_78]], %[[VAL_77]][7, 1, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           %[[VAL_80:.*]] = llvm.mul %[[VAL_66]], %[[VAL_20_EX1]]  : i64
// CHECK:           %[[VAL_81:.*]] = llvm.mul %[[VAL_67]], %[[VAL_20_EX1]]  : i64
// CHECK:           %[[VAL_82:.*]] = llvm.getelementptr %[[VAL_24_BASEPTR]]{{\[}}%[[VAL_70_OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:           %[[VAL_84:.*]] = llvm.insertvalue %[[VAL_82]], %[[VAL_79]][0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>
// CHECK:           ptr.store %[[VAL_84]], %[[VAL_8]] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<2 x array<3 x i64>>)>, !llvm.ptr
// CHECK:           llvm.return
// CHECK:         }
func.func @test_char1(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %3 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>>
  %4 = fir.box_elesize %3 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> index
  %5:3 = fir.box_dims %3, %c1 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>, index) -> (index, index, index)
  %8 = fir.box_addr %3 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x?x!fir.char<1,?>>>
  %9:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,?>>>>, index) -> (index, index, index)
  %10 = arith.divsi %4, %c1 : index
  %12 = fircg.ext_embox %8(%9#1, %5#1) origin %9#0, %5#0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6] typeparams %10 : (!fir.heap<!fir.array<?x?x!fir.char<1,?>>>, index, index, index, index, index, index, index, index, index, index, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  return
}
