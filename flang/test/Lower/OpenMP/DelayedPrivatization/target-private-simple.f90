! Tests delayed privatization for `targets ... private(..)` for simple variables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine target_simple
  implicit none
  integer :: simple_var

  !$omp target private(simple_var)
    simple_var = 10
  !$omp end target
end subroutine target_simple

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME:              @[[VAR_PRIVATIZER_SYM:.*]] : i32

! CHECK-LABEL: func.func @_QPtarget_simple() {
! CHECK:  %[[VAR_ALLOC:.*]] = fir.alloca i32 {bindc_name = "simple_var", {{.*}}}
! CHECK:  %[[VAR_DECL:.*]]:2 = hlfir.declare %[[VAR_ALLOC]]

! CHECK:  omp.target private(
! CHECK-SAME: @[[VAR_PRIVATIZER_SYM]] %[[VAR_DECL]]#0 -> %[[REG_ARG:.*]] : !fir.ref<i32>) {
! CHECK:      %[[REG_DECL:.*]]:2 = hlfir.declare %[[REG_ARG]]
! CHECK:      %[[C10:.*]] = arith.constant 10
! CHECK:      hlfir.assign %[[C10]] to %[[REG_DECL]]#0
! CHECK:      omp.terminator
! CHECK:    }

! CHECK:    return
! CHECK:  }

