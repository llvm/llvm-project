! RUN: %flang_fc1 -fopenacc -fdebug-unparse %s | FileCheck %s

! Test unparse does not crash with OpenACC directives.

! Test bug 47659
program bug47659
  integer :: i, j
  label1: do i = 1, 10
    !$acc parallel loop
    do j = 1, 10
      if (j == 2) then
        stop 1
      end if
    end do
  end do label1
end program

!CHECK-LABEL: PROGRAM BUG47659
!CHECK: !$ACC PARALLEL LOOP


subroutine acc_loop()
  integer :: i, j
  real :: a(10)
  integer :: gangNum, gangDim, gangStatic

!CHECK-LABEL: SUBROUTINE acc_loop

  !$acc loop collapse(force: 2)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
!CHECK: !$ACC LOOP COLLAPSE(FORCE:2_4)

! Blanks are permitted around the ':' separator of a collapse force modifier.
  !$acc loop collapse(force : 2)
  do i = 1, 10
    do j = 1, 10
    end do
  end do
!CHECK: !$ACC LOOP COLLAPSE(FORCE:2_4)

  !$acc loop gang
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG

  !$acc loop gang(gangNum)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(NUM:gangnum)

  !$acc loop gang(num: gangNum)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(NUM:gangnum)

  !$acc loop gang(dim: gangDim)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(DIM:gangdim)

  !$acc loop gang(static:gangStatic)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:gangstatic)

  !$acc loop gang(static:*)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:*)

  !$acc loop gang(static:gangStatic, dim: gangDim)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:gangstatic,DIM:gangdim)

! Spaces are permitted around the ':' separator of a gang-arg.
  !$acc loop gang(static :gangStatic)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:gangstatic)

  !$acc loop gang(static : gangStatic)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:gangstatic)

  !$acc loop gang(static : *)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(STATIC:*)

  !$acc loop gang(dim : gangDim)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(DIM:gangdim)

  !$acc loop gang(num : gangNum)
  do i = 1, 10
    a(i) = i
  end do
! CHECK: !$ACC LOOP GANG(NUM:gangnum)

end subroutine

! Blanks are permitted around the ':' separators of a wait-argument
! (devnum/queues) and of copy/create data modifiers (zero/readonly).
subroutine acc_clause_spaces()
  integer :: i
  real :: a(10), b(10)

!CHECK-LABEL: SUBROUTINE acc_clause_spaces

  !$acc wait(devnum : 1 : 2, 3)
!CHECK: !$ACC WAIT(DEVNUM:1_4:2_4,3_4)

  !$acc data copyin(readonly : a) create(zero : b)
!CHECK: !$ACC DATA COPYIN(READONLY:a) CREATE(ZERO:b)
  !$acc end data
!CHECK: !$ACC END DATA
end subroutine

subroutine routine1()
  !$acc routine bind("routine1_")
! CHECK: !$ACC ROUTINE BIND("routine1_")
end subroutine

subroutine routine2()
  !$acc routine(routine2) bind(routine2)
! CHECK: !$ACC ROUTINE(routine2) BIND(routine2)
end subroutine

subroutine routine3()
end subroutine

module routine_multi_mod
  ! Multi-name form: round-trips as-is (NV extension, not canonicalized).
  !$acc routine(routine2, routine3) seq
! CHECK: !$ACC ROUTINE(routine2,routine3) SEQ
end module
