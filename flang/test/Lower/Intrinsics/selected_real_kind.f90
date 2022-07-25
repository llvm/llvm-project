! REQUIRES: shell
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPselected_real_kind_test1(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i8> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i8> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i8 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test1Eres"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i8
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i8>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test1(p, r, d)
  integer(1) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test2(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i16> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i16> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i16 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test2Eres"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i16
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i16>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test2(p, r, d)
  integer(2) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test4(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test4Eres"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test4(p, r, d)
  integer(4) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test8(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "p"},
! CHECK-SAME:                                         %[[VAL_1:.*]]: !fir.ref<i64> {fir.bindc_name = "r"},
! CHECK-SAME:                                         %[[VAL_2:.*]]: !fir.ref<i64> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i64 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test8Eres"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 8 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 8 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 8 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test8(p, r, d)
  integer(8) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test16(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<i128> {fir.bindc_name = "p"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<i128> {fir.bindc_name = "r"},
! CHECK-SAME:                                          %[[VAL_2:.*]]: !fir.ref<i128> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca i128 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test16Eres"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 16 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 16 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 16 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i128
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<i128>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test16(p, r, d)
  integer(16) :: p, r, d, res
  res = selected_real_kind(P=p, R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_rd(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "r"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test_rdEres"}
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.ref<i1>
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i1>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_rd(r, d)
  integer :: r, d, res
  res = selected_real_kind(R=r, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_pd(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test_pdEres"}
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.ref<i1>
! CHECK:         %[[VAL_6:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i1>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_pd(p, d)
  integer :: p, d, res
  res = selected_real_kind(P=p, RADIX=d)
end

! CHECK-LABEL: func.func @_QPselected_real_kind_test_pr(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "p"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "r"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_real_kind_test_prEres"}
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.ref<i1>
! CHECK:         %[[VAL_6:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_7:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i1>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranASelectedRealKind(%{{.*}}, %{{.*}}, %[[VAL_10]], %[[VAL_6]], %[[VAL_11]], %[[VAL_7]], %[[VAL_12]], %[[VAL_8]]) : (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_real_kind_test_pr(p, r)
  integer :: p, r, res
  res = selected_real_kind(P=p, R=r)
end
