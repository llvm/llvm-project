! RUN: %flang_fc1 -fopenmp -emit-fir %s -o - | FileCheck %s

! Check that this testcase is lowered to FIR successfully.

! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
! CHECK: %[[DECL_N:.*]] = fir.declare %{{.*}} {uniq_name = "_QMtestEn"} : (!fir.ref<i64>) -> !fir.ref<i64>
! CHECK: %[[HOST_N:.*]] = fir.load %[[DECL_N]] : !fir.ref<i64>
! CHECK: %[[HOST_LB:.*]] = fir.convert %[[ONE]] : (i32) -> i64
! CHECK: %[[HOST_STEP:.*]] = fir.convert %[[ONE]] : (i32) -> i64
! CHECK:      omp.target
! CHECK-SAME: host_eval(%[[HOST_LB]] -> %[[LB:[[:alnum:]]+]], %[[HOST_N]] -> %[[UB:[[:alnum:]]+]], %[[HOST_STEP]] -> %[[STEP:[[:alnum:]]+]] : i64, i64, i64)
! CHECK:      omp.teams
! CHECK:      omp.parallel
! CHECK:      omp.distribute
! CHECK-NEXT: omp.wsloop
! CHECK-NEXT: omp.loop_nest ({{.*}}) : i64 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])

module Test
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64
    implicit none
    integer(kind=INT64) :: N
    real(kind=REAL64), allocatable :: A(:)

    contains
        subroutine init_arrays(initA)
            implicit none
            real(kind=REAL64), intent(in) :: initA
            integer(kind=INT64) :: i
            !$omp target teams distribute parallel do
            do i = 1, N
                A(i) = initA
            end do
        end subroutine init_arrays

end module Test
