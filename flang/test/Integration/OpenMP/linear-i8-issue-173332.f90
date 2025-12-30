! Integration test to verify INTEGER(8) linear clause compiles successfully
! This is a regression test for issue #173332
! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o - | FileCheck %s

subroutine repro_issue_173332
    implicit none
    integer(8) :: i, j

    j = 0
    ! This used to fail with:
    ! "Cannot create binary operator with two operands of differing type!"
    ! The fix ensures all arithmetic is normalized to the linear variable's type (i64)
    !$omp parallel do simd linear(j)
    do i = 1,100,1
        j = j + 1
    end do
    !$omp end parallel do simd
end subroutine repro_issue_173332

! CHECK-LABEL: define {{.*}} @repro_issue_173332_
! CHECK: ret void
