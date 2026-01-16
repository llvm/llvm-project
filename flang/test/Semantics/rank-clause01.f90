! Test the new RANK clause.  This uses the examples from the F2023 Standard and
! related explanation documents.
!
! RUN: %flang %s -o %t
! RUN: env LD_LIBRARY_PATH="$LD_LIBRARY_PATH:%libdir" %t | FileCheck %s
! CHECK-NOT: FAIL

program rank_clause01
    implicit none

    logical :: X0(10,10,10)
    integer :: array1(10,10)

    interface

      subroutine sub02(arg1)
        integer, rank(2) :: arg1
      end subroutine

      subroutine new(arg1)
        integer, rank(2) :: arg1
      end subroutine

    end interface

    call sub01(X0)

    call sub02(array1)

    call new(array1)

  contains

    subroutine sub01(X3)

      integer :: X0(10,10,10)
      logical, rank(rank(X0)), allocatable :: X1 ! Rank 3, deferred shape
      complex, rank(2), pointer :: X2 ! Rank 2, deferred-shape
      logical, rank(rank(X0)) :: X3 ! Rank 3, assumed-shape
      real, rank(0) :: X4 ! Scalar
      allocatable :: X4

      if (rank(X1) == 3 .and. rank(X2) == 2 .and. rank(X3) == 3 .and. &
          rank(X4) == 0) then
        print *, "PASS"
      else
        print *, "FAIL"
      endif

    end subroutine

end program

subroutine sub02(A)
    integer, rank(2) :: A, B
    entry new(B)

    if (rank(A) == rank(B)) then
      print *, "PASS"
    else
      print *, "FAIL"
    endif

end subroutine
