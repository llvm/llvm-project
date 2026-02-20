// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// derived from:
// subroutine ido3
//   integer :: j
//   write(*,*,err=404) (j,j=1,10)
// 404 continue
// end subroutine

// CHECK-LABEL: func.func @iterate_while
// CHECK:       scf.while (%[[IV:.*]] = %c1, %[[OK:.*]] = %true) : (index, i1) -> (index, i1) {
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[CMP0:.*]] = arith.cmpi slt, %[[C0]], %c1 : index
// CHECK:         %[[CMP1:.*]] = arith.cmpi sle, %[[IV]], %c10 : index
// CHECK:         %[[CMP2:.*]] = arith.cmpi slt, %c1, %[[C0]] : index
// CHECK:         %[[CMP3:.*]] = arith.cmpi sge, %[[IV]], %c10 : index
// CHECK:         %[[AND0:.*]] = arith.andi %[[CMP0]], %[[CMP1]] : i1
// CHECK:         %[[AND1:.*]] = arith.andi %[[CMP2]], %[[CMP3]] : i1
// CHECK:         %[[OR:.*]] = arith.ori %[[AND0]], %[[AND1]] : i1
// CHECK:         %[[AND2:.*]] = arith.andi %[[OK]], %[[OR]] : i1
// CHECK:         scf.condition(%[[AND2]]) %[[IV]], %[[OK]] : index, i1
// CHECK:       } do {
// CHECK:       ^bb0(%[[IV2:.*]]: index, %[[OK2:.*]]: i1):
// CHECK:         %[[NEXT:.*]] = arith.addi %[[IV2]], %c1
// CHECK:         scf.if %[[OK2]]
// CHECK:         scf.yield

func.func @iterate_while() {
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %false = arith.constant false
  %c4_i32 = arith.constant 4 : i32
  %c6_i32 = arith.constant 6 : i32
  %0 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFido3Ej"}
  %j = fir.declare %0 {uniq_name = "_QFido3Ej"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.address_of(@_QQclX16525d77bf698d01a3a03f9bc4435ebe) : !fir.ref<!fir.char<1,45>>
  %3 = fir.convert %2 : (!fir.ref<!fir.char<1,45>>) -> !fir.ref<i8>
  %4 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %3, %c4_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %5 = fir.call @_FortranAioEnableHandlers(%4, %false, %true, %false, %false, %false) fastmath<contract> : (!fir.ref<i8>, i1, i1, i1, i1, i1) -> none
  %6:2 = fir.iterate_while (%arg0 = %c1 to %c10 step %c1) and (%arg1 = %true) -> (index, i1) {
    %10 = arith.index_cast %arg0 : index to i32
    fir.store %10 to %j : !fir.ref<i32>
    %11 = fir.if %arg1 -> (i1) {
      %14 = fir.load %j : !fir.ref<i32>
      %15 = fir.call @_FortranAioOutputInteger32(%4, %14) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
      fir.result %15 : i1
    } else {
      fir.result %false : i1
    }
    %12 = arith.addi %arg0, %c1 overflow<nsw> : index
    %13 = arith.select %11, %12, %arg0 : index
    fir.result %13, %11 : index, i1
  }
  %7 = arith.index_cast %6#0 : index to i32
  fir.store %7 to %j : !fir.ref<i32>
  %8 = fir.call @_FortranAioEndIoStatement(%4) fastmath<contract> : (!fir.ref<i8>) -> i32
  %9 = arith.index_cast %8 : i32 to index
  fir.select %9 : index [0, ^bb1, unit, ^bb1]
^bb1:  // 2 preds: ^bb0, ^bb0
  return
}

// Check if iv is added to scf.while result, and if the normalized loop will
// return adjusted_iv+offset.
//
// derived from:
// subroutine sub1(string)
//  implicit none
//  character*(*) string
//  print *, len_trim(string)
// end subroutine

// CHECK-LABEL: func.func @normalize
// CHECK:       [[WHILE:%[0-9]+]]:3 = scf.while (%[[IVIN:.*]] = %7, %[[OKIN:.*]] = %true, %[[IV2IN:.*]] = %7) : (index, i1, index) -> (index, i1, index) {
// CHECK:         [[C0:%.*]] = arith.constant 0 : index
// CHECK:         [[CMP0:%[0-9]+]] = arith.cmpi slt, [[C0]], %c-1 : index
// CHECK:         [[CMP1:%[0-9]+]] = arith.cmpi sle, %[[IVIN]], %c0 : index
// CHECK:         [[CMP2:%[0-9]+]] = arith.cmpi slt, %c-1, [[C0]] : index
// CHECK:         [[CMP3:%[0-9]+]] = arith.cmpi sge, %[[IVIN]], %c0 : index
// CHECK:         [[AND0:%[0-9]+]] = arith.andi [[CMP0]], [[CMP1]] : i1
// CHECK:         [[AND1:%[0-9]+]] = arith.andi [[CMP2]], [[CMP3]] : i1
// CHECK:         [[OR:%[0-9]+]] = arith.ori [[AND0]], [[AND1]] : i1
// CHECK:         [[AND2:%[0-9]+]] = arith.andi %[[OKIN]], [[OR]] : i1
// CHECK:         scf.condition([[AND2]]) %[[IVIN]], %[[OKIN]], %[[IV2IN]] : index, i1, index
// CHECK:       } do {
// CHECK:       ^bb0(%[[IV:.*]]: index, %[[OK:.*]]: i1, %[[IV2:.*]]: index):
// CHECK:         [[NEXT:%[0-9]+]] = arith.addi %[[IV]], %c-1 : index
// CHECK:         [[ISSPACE:%[0-9]+]] = arith.cmpi eq, %{{.*}}, %c32_i8 : i8
// CHECK:         scf.yield [[NEXT]], [[ISSPACE]], %[[IV]] : index, i1, index

func.func @normalize(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "string"}, %arg1: i64) {
  %0 = fir.emboxchar %arg0, %arg1 : (!fir.ref<!fir.char<1,?>>, i64) -> !fir.boxchar<1>
  %c32_i8 = arith.constant 32 : i8
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %c5_i32 = arith.constant 5 : i32
  %c6_i32 = arith.constant 6 : i32
  %1 = fir.undefined !fir.dscope
  %2:2 = fir.unboxchar %0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3 = fir.declare %2#0 typeparams %2#1 dummy_scope %1 {uniq_name = "_QFsub1Estring"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>>
  %4 = fir.address_of(@_QQclX9c7b903babe9837742def6df4989c6e7) : !fir.ref<!fir.char<1,40>>
  %5 = fir.convert %4 : (!fir.ref<!fir.char<1,40>>) -> !fir.ref<i8>
  %6 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %5, %c5_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %7 = arith.subi %2#1, %c1 : index
  %8:2 = fir.iterate_while (%arg2 = %7 to %c0 step %c-1) and (%arg3 = %true) iter_args(%arg4 = %7) -> (index) {
    %14 = fir.convert %3 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    %15 = fir.coordinate_of %14, %arg2 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    %16 = fir.convert %15 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
    %17 = fir.load %16 : !fir.ref<i8>
    %18 = arith.cmpi eq, %17, %c32_i8 : i8
    fir.result %18, %arg2 : i1, index
  }
  %9 = arith.addi %8#1, %c1 : index
  %10 = arith.select %8#0, %c0, %9 : index
  %11 = arith.index_cast %10 : index to i32
  %12 = fir.call @_FortranAioOutputInteger32(%6, %11) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
  %13 = fir.call @_FortranAioEndIoStatement(%6) fastmath<contract> : (!fir.ref<i8>) -> i32
  return
}


