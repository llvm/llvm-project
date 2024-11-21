! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPmalloc_ptr() {
subroutine malloc_ptr()
  integer :: x
  pointer (ptr_x, x)
  ! CHECK:           %[[X:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
  ! CHECK:           %[[X_PTR:.*]] = fir.alloca i64 {bindc_name = "ptr_x", uniq_name = "_QFmalloc_ptrEptr_x"}
  ! CHECK:           %[[X_PTR_DECL:.*]]:2 = hlfir.declare %[[X_PTR]] {uniq_name = "_QFmalloc_ptrEptr_x"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  ! CHECK:           %[[CST:.*]] = arith.constant 4 : i32
  ! CHECK:           %[[CST_I64:.*]] = fir.convert %[[CST]] : (i32) -> i64
  ! CHECK:           %[[ALLOC:.*]] = fir.call @_FortranAMalloc(%[[CST_I64]]) fastmath<contract> : (i64) -> i64
  ! CHECK:           hlfir.assign %[[ALLOC]] to %[[X_PTR_DECL]]#0 : i64, !fir.ref<i64>
  ! CHECK:           return
  ptr_x = malloc(4)
end subroutine

! gfortran allows malloc to be assigned to integers, so we accept it.

! CHECK-LABEL:   func.func @_QPmalloc_i8() {
subroutine malloc_i8()
  integer(kind=1) :: x
! CHECK:           %[[X:.*]] = fir.alloca i8 {bindc_name = "x", uniq_name = "_QFmalloc_i8Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFmalloc_i8Ex"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK:           %[[CST:.*]] = arith.constant 1 : i32
! CHECK:           %[[CST_I64:.*]] = fir.convert %[[CST]] : (i32) -> i64
! CHECK:           %[[ALLOC:.*]] = fir.call @_FortranAMalloc(%[[CST_I64]]) fastmath<contract> : (i64) -> i64
! CHECK:           %[[ALLOC_I8:.*]] = fir.convert %[[ALLOC]] : (i64) -> i8
! CHECK:           hlfir.assign %[[ALLOC_I8]] to %[[X_DECL]]#0 : i8, !fir.ref<i8>
! CHECK:           return
  x = malloc(1)
end subroutine

! CHECK-LABEL:   func.func @_QPmalloc_i16() {
subroutine malloc_i16()
  integer(kind=2) :: x
! CHECK:           %[[X:.*]] = fir.alloca i16 {bindc_name = "x", uniq_name = "_QFmalloc_i16Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFmalloc_i16Ex"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK:           %[[CST:.*]] = arith.constant 1 : i32
! CHECK:           %[[CST_I64:.*]] = fir.convert %[[CST]] : (i32) -> i64
! CHECK:           %[[ALLOC:.*]] = fir.call @_FortranAMalloc(%[[CST_I64]]) fastmath<contract> : (i64) -> i64
! CHECK:           %[[ALLOC_I16:.*]] = fir.convert %[[ALLOC]] : (i64) -> i16
! CHECK:           hlfir.assign %[[ALLOC_I16]] to %[[X_DECL]]#0 : i16, !fir.ref<i16>
! CHECK:           return
  x = malloc(1)
end subroutine


! CHECK-LABEL:   func.func @_QPmalloc_i32() {
subroutine malloc_i32()
  integer(kind=4) :: x
! CHECK:           %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmalloc_i32Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFmalloc_i32Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[CST:.*]] = arith.constant 1 : i32
! CHECK:           %[[CST_I64:.*]] = fir.convert %[[CST]] : (i32) -> i64
! CHECK:           %[[ALLOC:.*]] = fir.call @_FortranAMalloc(%[[CST_I64]]) fastmath<contract> : (i64) -> i64
! CHECK:           %[[ALLOC_I32:.*]] = fir.convert %[[ALLOC]] : (i64) -> i32
! CHECK:           hlfir.assign %[[ALLOC_I32]] to %[[X_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           return
  x = malloc(1)
end subroutine

! CHECK-LABEL:   func.func @_QPmalloc_i64() {
subroutine malloc_i64()
  integer(kind=8) :: x
! CHECK:           %[[X:.*]] = fir.alloca i64 {bindc_name = "x", uniq_name = "_QFmalloc_i64Ex"}
! CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFmalloc_i64Ex"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[CST:.*]] = arith.constant 1 : i32
! CHECK:           %[[CST_I64:.*]] = fir.convert %[[CST]] : (i32) -> i64
! CHECK:           %[[ALLOC:.*]] = fir.call @_FortranAMalloc(%[[CST_I64]]) fastmath<contract> : (i64) -> i64
! CHECK:           hlfir.assign %[[ALLOC]] to %[[X_DECL]]#0 : i64, !fir.ref<i64>
! CHECK:           return
  x = malloc(1)
end subroutine
