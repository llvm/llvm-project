! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine selected_logical_kind_test1(input)
  integer(1) :: input, res
  res = selected_logical_kind(input)
end

! CHECK-LABEL: func.func @_QPselected_logical_kind_test1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i8> {fir.bindc_name = "input"})
! CHECK: %[[INPUT:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {uniq_name = "_QFselected_logical_kind_test1Einput"} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK: %[[RES_ALLOCA:.*]] = fir.alloca i8 {bindc_name = "res", uniq_name = "_QFselected_logical_kind_test1Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFselected_logical_kind_test1Eres"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK: %[[KIND:.*]] = arith.constant 1 : i32
! CHECK: %[[INPUT_ADDR:.*]] = fir.convert %1#1 : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK: %{{.*}} = fir.call @_FortranASelectedLogicalKind(%{{.*}}, %{{.*}}, %[[INPUT_ADDR]], %[[KIND]]) fastmath<contract> : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32

subroutine selected_logical_kind_test2(input)
  integer(2) :: input, res
  res = selected_logical_kind(input)
end

! CHECK-LABEL: func.func @_QPselected_logical_kind_test2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i16> {fir.bindc_name = "input"})
! CHECK: %[[INPUT:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {uniq_name = "_QFselected_logical_kind_test2Einput"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK: %[[RES_ALLOCA:.*]] = fir.alloca i16 {bindc_name = "res", uniq_name = "_QFselected_logical_kind_test2Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFselected_logical_kind_test2Eres"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK: %[[KIND:.*]] = arith.constant 2 : i32
! CHECK: %[[INPUT_ADDR:.*]] = fir.convert %[[INPUT]]#1 : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK: %{{.*}} = fir.call @_FortranASelectedLogicalKind(%{{.*}}, %{{.*}}, %[[INPUT_ADDR]], %[[KIND]]) fastmath<contract> : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32

subroutine selected_logical_kind_test4(input)
  integer(4) :: input, res
  res = selected_logical_kind(input)
end

! CHECK-LABEL: func.func @_QPselected_logical_kind_test4(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "input"})
! CHECK: %[[INPUT:.*]]:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFselected_logical_kind_test4Einput"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[RES_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_logical_kind_test4Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFselected_logical_kind_test4Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[KIND:.*]] = arith.constant 4 : i32
! CHECK: %[[INPUT_ADDR:.*]] = fir.convert %[[INPUT]]#1 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK: %{{.*}} = fir.call @_FortranASelectedLogicalKind(%{{.*}}, %{{.*}}, %[[INPUT_ADDR]], %[[KIND]]) fastmath<contract> : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32

subroutine selected_logical_kind_test8(input)
  integer(8) :: input, res
  res = selected_logical_kind(input)
end

! CHECK-LABEL: func.func @_QPselected_logical_kind_test8(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i64> {fir.bindc_name = "input"})
! CHECK: %[[INPUT:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {uniq_name = "_QFselected_logical_kind_test8Einput"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK: %[[RES_ALLOCA]] = fir.alloca i64 {bindc_name = "res", uniq_name = "_QFselected_logical_kind_test8Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFselected_logical_kind_test8Eres"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK: %[[KIND:.*]] = arith.constant 8 : i32
! CHECK: %[[INPUT_ADDR:.*]] = fir.convert %[[INPUT]]#1 : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK: %{{.*}} = fir.call @_FortranASelectedLogicalKind(%{{.*}}, %{{.*}}, %[[INPUT_ADDR]], %[[KIND]]) fastmath<contract> : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32

subroutine selected_logical_kind_test16(input)
  integer(16) :: input, res
  res = selected_logical_kind(input)
end

! CHECK-LABEL: func.func @_QPselected_logical_kind_test16(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i128> {fir.bindc_name = "input"})
! CHECK: %[[INPUT:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {uniq_name = "_QFselected_logical_kind_test16Einput"} : (!fir.ref<i128>, !fir.dscope) -> (!fir.ref<i128>, !fir.ref<i128>)
! CHECK: %[[RES_ALLOCA:.*]] = fir.alloca i128 {bindc_name = "res", uniq_name = "_QFselected_logical_kind_test16Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {uniq_name = "_QFselected_logical_kind_test16Eres"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
! CHECK: %[[KIND:.*]] = arith.constant 16 : i32
! CHECK: %[[INPUT_ADDR:.*]] = fir.convert %[[INPUT]]#1 : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK: %{{.*}} = fir.call @_FortranASelectedLogicalKind(%{{.*}}, %{{.*}}, %[[INPUT_ADDR]], %[[KIND]]) fastmath<contract> : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
