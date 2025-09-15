! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -finit-global-zero -emit-hlfir -o - %s | FileCheck %s
! RUN: %flang_fc1 -fno-init-global-zero -emit-hlfir -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
! RUN: bbc -finit-global-zero -emit-hlfir -o - %s | FileCheck %s
! RUN: bbc -finit-global-zero=false -emit-hlfir -o - %s | FileCheck %s

! Test that the flag does not affect globals with default init

module zeroInitM2
  type val
    integer :: my_val = 1
  end type val
  type(val) :: v1
end module zeroInitM2

!CHECK:  fir.global @_QMzeroinitm2Ev1 : !fir.type<_QMzeroinitm2Tval{my_val:i32}> {
!CHECK:    %[[V1:.*]] = fir.undefined !fir.type<_QMzeroinitm2Tval{my_val:i32}>
!CHECK:    %[[ONE:.*]] = arith.constant 1 : i32
!CHECK:    %[[V1_INIT:.*]] = fir.insert_value %[[V1]], %[[ONE]], ["my_val", !fir.type<_QMzeroinitm2Tval{my_val:i32}>] : (!fir.type<_QMzeroinitm2Tval{my_val:i32}>, i32) -> !fir.type<_QMzeroinitm2Tval{my_val:i32}>
!CHECK:    fir.has_value %[[V1_INIT]] : !fir.type<_QMzeroinitm2Tval{my_val:i32}>
!CHECK:  }
