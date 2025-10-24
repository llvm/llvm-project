! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPselected_int_kind_test1(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i8 {bindc_name = "res", uniq_name = "_QFselected_int_kind_test1Eres"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i8>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[VAL_6]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i8
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i8>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test1(a)
  integer(1) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test2(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i16 {bindc_name = "res", uniq_name = "_QFselected_int_kind_test2Eres"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i16>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[VAL_6]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i16
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i16>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test2(a)
  integer(2) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test4(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFselected_int_kind_test4Eres"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[VAL_6]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test4(a)
  integer(4) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test8(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i64 {bindc_name = "res", uniq_name = "_QFselected_int_kind_test8Eres"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 8 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i64>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[VAL_6]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i64
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i64>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test8(a)
  integer(8) :: a, res
  res = selected_int_kind(a)
end

! CHECK-LABEL: func.func @_QPselected_int_kind_test16(
! CHECK-SAME:                                         %[[VAL_0:.*]]: !fir.ref<i128> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i128 {bindc_name = "res", uniq_name = "_QFselected_int_kind_test16Eres"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 16 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<i128>) -> !fir.llvm_ptr<i8>
! CHECK:         %[[VAL_7:.*]] = fir.call @_FortranASelectedIntKind(%{{.*}}, %{{.*}}, %[[VAL_6]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, i32, !fir.llvm_ptr<i8>, i32) -> i32
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> i128
! CHECK:         fir.store %[[VAL_8]] to %[[VAL_1]] : !fir.ref<i128>
! CHECK:         return
! CHECK:       }

subroutine selected_int_kind_test16(a)
  integer(16) :: a, res
  res = selected_int_kind(a)
end
