! This test checks lowering of stop statement in OpenACC region.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenacc %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_stop_in_region1() {
! CHECK:         acc.parallel {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant false
! CHECK:           %[[VAL_2:.*]] = arith.constant false
! CHECK:           %[[VAL_3:.*]] = fir.call @_FortranAStopStatement(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {{.*}} : (i32, i1, i1) -> none
! CHECK:           acc.yield
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region1()
  !$acc parallel
    stop 1
  !$acc end parallel
end

! CHECK-LABEL: func.func @_QPtest_stop_in_region2() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtest_stop_in_region2Ex"}
! CHECK:         acc.parallel {
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_2:.*]] = arith.constant false
! CHECK:           %[[VAL_3:.*]] = arith.constant false
! CHECK:           %[[VAL_4:.*]] = fir.call @_FortranAStopStatement(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]]) {{.*}} : (i32, i1, i1) -> none
! CHECK:           acc.yield
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_stop_in_region2()
  integer :: x
  !$acc parallel
    stop 1
    x = 2
  !$acc end parallel
end
