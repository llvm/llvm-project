! RUN: %flang_fc1 -fopenacc -fdebug-unparse %s | FileCheck %s

! Test unparse does not crash with OpenACC directives.

! Test bug 47659
program bug47659
  integer :: i, j
  label1: do i = 1, 10
    !$acc parallel loop
    do j = 1, 10
      if (j == 2) then
        exit label1
      end if
    end do
  end do label1
end program

!CHECK-LABEL: PROGRAM bug47659
!CHECK: !$ACC PARALLEL LOOP


subroutine acc_loop()
  integer :: i, j

  !$acc loop collapse(force: 2)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
end subroutine

!CHECK-LABEL: SUBROUTINE acc_loop
!CHECK: !$ACC LOOP COLLAPSE(FORCE:2_4)
