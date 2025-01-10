! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:   func.func @_QPfree_ptr() {
subroutine free_ptr()
  integer :: x
  pointer (ptr_x, x)
  ! CHECK:           %[[X:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
  ! CHECK:           %[[X_PTR:.*]] = fir.alloca i64 {bindc_name = "ptr_x", uniq_name = "_QFfree_ptrEptr_x"}
  ! CHECK:           %[[X_PTR_DECL:.*]]:2 = hlfir.declare %[[X_PTR]] {uniq_name = "_QFfree_ptrEptr_x"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFfree_ptrEx"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  ! CHECK:           %[[X_LD:.*]] = fir.load %[[X_PTR_DECL]]#0 : !fir.ref<i64>
  ! CHECK:           %[[VOID:.*]] = fir.call @_FortranAFree(%[[X_LD]]) fastmath<contract> : (i64) -> none
  ! CHECK:           return
  call free(ptr_x)
end subroutine

! gfortran allows free to be used on integers, so we accept it with a warning.

! CHECK-LABEL:   func.func @_QPfree_i8() {
subroutine free_i8
  integer (kind=1) :: x
  ! CHECK:           %[[X:.*]] = fir.alloca i8 {bindc_name = "x", uniq_name = "_QFfree_i8Ex"}
  ! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFfree_i8Ex"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
  ! CHECK:           %[[X_LD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i8>
  ! CHECK:           %[[X_I64:.*]] = fir.convert %[[X_LD]] : (i8) -> i64
  ! CHECK:           %[[VOID:.*]] = fir.call @_FortranAFree(%[[X_I64]]) fastmath<contract> : (i64) -> none
  ! CHECK:           return
  call free(x)
end subroutine


! CHECK-LABEL:   func.func @_QPfree_i16() {
subroutine free_i16
  integer (kind=2) :: x
  ! CHECK:           %[[X:.*]] = fir.alloca i16 {bindc_name = "x", uniq_name = "_QFfree_i16Ex"}
  ! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFfree_i16Ex"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
  ! CHECK:           %[[X_LD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i16>
  ! CHECK:           %[[X_I64:.*]] = fir.convert %[[X_LD]] : (i16) -> i64
  ! CHECK:           %[[VOID:.*]] = fir.call @_FortranAFree(%[[X_I64]]) fastmath<contract> : (i64) -> none
  ! CHECK:           return
  call free(x)
end subroutine

! CHECK-LABEL:   func.func @_QPfree_i32() {
subroutine free_i32
  integer (kind=4) :: x
  ! CHECK:           %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFfree_i32Ex"}
  ! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFfree_i32Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK:           %[[X_LD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
  ! CHECK:           %[[X_I64:.*]] = fir.convert %[[X_LD]] : (i32) -> i64
  ! CHECK:           %[[VOID:.*]] = fir.call @_FortranAFree(%[[X_I64]]) fastmath<contract> : (i64) -> none
  ! CHECK:           return
  call free(x)
end subroutine

! CHECK-LABEL:   func.func @_QPfree_i64() {
subroutine free_i64
  integer (kind=8) :: x
  ! CHECK:           %[[X:.*]] = fir.alloca i64 {bindc_name = "x", uniq_name = "_QFfree_i64Ex"}
  ! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFfree_i64Ex"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK:           %[[X_LD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i64>
  ! CHECK:           %[[VOID:.*]] = fir.call @_FortranAFree(%[[X_LD]]) fastmath<contract> : (i64) -> none
  ! CHECK:           return
  call free(x)
end subroutine
