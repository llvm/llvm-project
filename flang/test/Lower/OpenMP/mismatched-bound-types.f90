! RUN: %flang_fc1 -fopenmp -emit-fir %s -o - | FileCheck %s

! Check that loops whose bounds have different integer types are lowered to FIR
! successfully. A literal bound is materialised at the loop variable's kind, so
! it needs no conversion; a bound of a different kind is converted.

module Test
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64
    implicit none
    integer(kind=INT64) :: N
    integer :: M
    real(kind=REAL64), allocatable :: A(:)

    contains
        ! A literal lower bound against an integer(8) upper bound.

        ! CHECK-LABEL: func.func @_QMtestPinit_arrays
        ! CHECK: %[[ONE:.*]] = arith.constant 1 : i64
        ! CHECK: %[[DECL_N:.*]] = fir.declare %{{.*}} {uniq_name = "_QMtestEn"} : (!fir.ref<i64>) -> !fir.ref<i64>
        ! CHECK: %[[HOST_N:.*]] = fir.load %[[DECL_N]] : !fir.ref<i64>
        ! CHECK:      omp.target
        ! CHECK-SAME: host_eval(%[[ONE]] -> %[[LB:[[:alnum:]]+]], %[[HOST_N]] -> %[[UB:[[:alnum:]]+]], %[[ONE]] -> %[[STEP:[[:alnum:]]+]] : i64, i64, i64)
        ! CHECK:      omp.teams
        ! CHECK:      omp.parallel
        ! CHECK:      omp.distribute
        ! CHECK-NEXT: omp.wsloop
        ! CHECK-NEXT: omp.loop_nest ({{.*}}) : i64 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
        subroutine init_arrays(initA)
            implicit none
            real(kind=REAL64), intent(in) :: initA
            integer(kind=INT64) :: i
            !$omp target teams distribute parallel do
            do i = 1, N
                A(i) = initA
            end do
        end subroutine init_arrays

        ! An integer(4) lower bound against an integer(8) upper bound, which
        ! does require a conversion.

        ! CHECK-LABEL: func.func @_QMtestPinit_arrays_mixed
        ! CHECK: %[[STEP2:.*]] = arith.constant 1 : i64
        ! CHECK: %[[DECL_M:.*]] = fir.declare %{{.*}} {uniq_name = "_QMtestEm"} : (!fir.ref<i32>) -> !fir.ref<i32>
        ! CHECK: %[[DECL_N2:.*]] = fir.declare %{{.*}} {uniq_name = "_QMtestEn"} : (!fir.ref<i64>) -> !fir.ref<i64>
        ! CHECK: %[[HOST_M:.*]] = fir.load %[[DECL_M]] : !fir.ref<i32>
        ! CHECK: %[[HOST_N2:.*]] = fir.load %[[DECL_N2]] : !fir.ref<i64>
        ! CHECK: %[[LB2_CONV:.*]] = fir.convert %[[HOST_M]] : (i32) -> i64
        ! CHECK:      omp.target
        ! CHECK-SAME: host_eval(%[[LB2_CONV]] -> %[[LB2:[[:alnum:]]+]], %[[HOST_N2]] -> %[[UB2:[[:alnum:]]+]], %[[STEP2]] -> %[[STEP2A:[[:alnum:]]+]] : i64, i64, i64)
        ! CHECK:      omp.teams
        ! CHECK:      omp.parallel
        ! CHECK:      omp.distribute
        ! CHECK-NEXT: omp.wsloop
        ! CHECK-NEXT: omp.loop_nest ({{.*}}) : i64 = (%[[LB2]]) to (%[[UB2]]) inclusive step (%[[STEP2A]])
        subroutine init_arrays_mixed(initA)
            implicit none
            real(kind=REAL64), intent(in) :: initA
            integer(kind=INT64) :: i
            !$omp target teams distribute parallel do
            do i = M, N
                A(i) = initA
            end do
        end subroutine init_arrays_mixed

end module Test
