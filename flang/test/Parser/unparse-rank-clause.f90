! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Test unparsing of RANK clause in array declaration statements

subroutine test_rank_clause(X3)
  integer :: X0(10,10,10)
  logical, rank(rank(X0)), allocatable :: X1 ! Rank 3, deferred shape
  complex, rank(2), pointer :: X2            ! Rank 2, deferred-shape
  logical, rank(rank(X0)) :: X3              ! Rank 3, assumed-shape
  real, allocatable, rank(0) :: X4           ! Scalar

  if (rank(X1) == 3 .and. rank(X2) == 2 .and. rank(X3) == 3 .and. &
      rank(X4) == 0) then
    print *, "PASS"
  else
    print *, "FAIL"
  endif

end subroutine

! CHECK: SUBROUTINE test_rank_clause (x3)
! CHECK:  INTEGER x0(10_4,10_4,10_4)
! CHECK:  LOGICAL, RANK(3_4), ALLOCATABLE :: x1
! CHECK:  COMPLEX, RANK(2_4), POINTER :: x2
! CHECK:  LOGICAL, RANK(3_4) :: x3
! CHECK:  REAL, ALLOCATABLE, RANK(0_4) :: x4
! CHECK:  IF (.true._4) THEN
! CHECK:   PRINT *, "PASS"
! CHECK:  ELSE
! CHECK:   PRINT *, "FAIL"
! CHECK:  END IF
! CHECK: END SUBROUTINE
