! RUN: %S/test_errors.sh %s %t %f18

!Tests for SELECT RANK Construct(R1148)
program select_rank
   implicit none
   integer, dimension(10:30, 10:20, -1:20) :: x
   integer, parameter :: y(*) = [1,2,3,4]
   integer, dimension(5) :: z
   integer, allocatable :: a(:)

   allocate(a(10:20))

   call CALL_SHAPE(x)
   call CALL_SHAPE(y)
   call CALL_SHAPE(z)
   call CALL_SHAPE(a)

contains
   !No error expected
   subroutine CALL_ME(x)
    implicit none
    integer :: x(..)
    SELECT RANK(x)
    RANK (0)
      print *, "PRINT RANK 0"
    RANK (1)
      print *, "PRINT RANK 1"
    END SELECT
   end

   subroutine CALL_ME9(x)
    implicit none
    integer :: x(..),j
    boo: SELECT RANK(x)
    RANK (1+0)
      print *, "PRINT RANK 1"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == (1+0)))
    END SELECT boo
   end subroutine

   !Error expected
   subroutine CALL_ME2(x)
    implicit none
    integer :: x(..)
    integer :: y(3),j
    !ERROR: Selector 'y' is not an assumed-rank array variable
    SELECT RANK(y)
    RANK (0)
      print *, "PRINT RANK 0"
    RANK (1)
      print *, "PRINT RANK 1"
     END SELECT

    SELECT RANK(x)
    RANK(0)
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 0)) ! will fail when RANK(x) is not zero here
    END SELECT
   end subroutine

   subroutine CALL_ME3(x)
    implicit none
    integer :: x(..),j
    SELECT RANK(x)
    !ERROR: The value of the selector must be between zero and 15
    RANK (16)
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 16))
    END SELECT
   end subroutine

   subroutine CALL_ME4(x)
    implicit none
    integer :: x(..)
    SELECT RANK(x)
    RANK DEFAULT
      print *, "ok "
    !ERROR: Not more than one of the selectors of SELECT RANK statement may be DEFAULT
    RANK DEFAULT
      print *, "not ok"
    RANK (3)
      print *, "IT'S 3"
    END SELECT
   end subroutine

   subroutine CALL_ME5(x)
    implicit none
    integer :: x(..),j
    SELECT RANK(x)
    RANK (0)
      print *, "PRINT RANK 0"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 0))
    RANK(1)
      print *, "PRINT RANK 1"
    !ERROR: Same rank value (0) not allowed more than once
    RANK(0)
      print *, "ERROR"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 0))
    RANK(1+1)
    !ERROR: Same rank value (2) not allowed more than once
    RANK(1+1)
    END SELECT
   end subroutine

   subroutine CALL_ME6(x)
    implicit none
    integer :: x(..),j
    SELECT RANK(x)
    RANK (3)
      print *, "one"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 3))
    !ERROR: The value of the selector must be between zero and 15
    RANK(-1)
      print *, "rank: -ve"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == -1))
    END SELECT
   end subroutine

   subroutine CALL_ME7(arg)
   implicit none
   integer :: i,j
   integer, dimension(..), pointer :: arg
   integer, pointer :: arg2
   !ERROR: RANK (*) cannot be used when selector is POINTER or ALLOCATABLE
   select RANK(arg)
   RANK (*)
      print *, arg(1:1)
   RANK (1)
      print *, arg
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(arg) == 1))
   end select

   !ERROR: Selector 'arg2' is not an assumed-rank array variable
   select RANK(arg2)
   RANK (*)
      print *,"This would lead to crash when saveSelSymbol has std::nullptr"
   RANK (1)
      print *, "Rank is 1"
   end select

   end subroutine

   subroutine CALL_ME8(x)
    implicit none
    integer :: x(..),j
    SELECT RANK(x)
    Rank(2)
      print *, "Now it's rank 2 "
    RANK (*)
      print *, "Going for a other rank"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 1))
    !ERROR: Not more than one of the selectors of SELECT RANK statement may be '*'
    RANK (*)
      print *, "This is Wrong"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 1))
    END SELECT
   end subroutine

   subroutine CALL_ME10(x)
    implicit none
    integer:: x(..), a=10,b=20,j
    integer, dimension(5) :: arr = (/1,2,3,4,5/),brr
    integer :: const_variable=10
    integer, pointer :: ptr,nullptr=>NULL()
    type derived
         character(len = 50) :: title
    end type derived
    type(derived) :: obj1

    SELECT RANK(x)
    Rank(2)
      print *, "Now it's rank 2 "
    RANK (*)
      print *, "Going for a other rank"
    !ERROR: Not more than one of the selectors of SELECT RANK statement may be '*'
    RANK (*)
      print *, "This is Wrong"
    END SELECT

    !ERROR: Selector 'brr' is not an assumed-rank array variable
    SELECT RANK(ptr=>brr)
    !ERROR: Must be a constant value
    RANK(const_variable)
      print *, "PRINT RANK 3"
      !j = INT(0, KIND=MERGE(KIND(0), -1, RANK(ptr) == 1))
    !ERROR: Must be a constant value
    RANK(nullptr)
      print *, "PRINT RANK 3"
    END SELECT

    !ERROR: Selector 'x(1) + x(2)' is not an assumed-rank array variable
    SELECT RANK (x(1) + x(2))

    END SELECT

    !ERROR: Selector 'x(1)' is not an assumed-rank array variable
    SELECT RANK(x(1))

    END SELECT

    !ERROR: Selector 'x(1:2)' is not an assumed-rank array variable
    SELECT RANK(x(1:2))

    END SELECT

    !ERROR: 'x' is not an object of derived type
    SELECT RANK(x(1)%x(2))

    END SELECT

    !ERROR: Selector 'obj1%title' is not an assumed-rank array variable
    SELECT RANK(obj1%title)

    END SELECT

    !ERROR: Selector 'arr(1:2)+ arr(4:5)' is not an assumed-rank array variable
    SELECT RANK(arr(1:2)+ arr(4:5))

    END SELECT

    SELECT RANK(ptr=>x)
    RANK (3)
      PRINT *, "PRINT RANK 3"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(ptr) == 0))
    RANK (1)
      PRINT *, "PRINT RANK 1"
      j = INT(0, KIND=MERGE(KIND(0), -1, RANK(ptr) == 1))
    END SELECT
   end subroutine
   subroutine CALL_ME_TYPES(x)
    implicit none
    integer :: x(..),j
    SELECT RANK(x)
    !ERROR: Must have INTEGER type, but is LOGICAL(4)
        RANK(.TRUE.)
    !ERROR: Must have INTEGER type, but is REAL(4)
        RANK(1.0)
    !ERROR: Must be a constant value
        RANK(RANK(x))
    !ERROR: Must have INTEGER type, but is CHARACTER(1)
        RANK("STRING")
    END SELECT
   end subroutine
   subroutine CALL_SHAPE(x)
    implicit none
    integer :: x(..)
    integer :: j
    integer, pointer :: ptr
    SELECT RANK(x)
     RANK(1)
       print *, "RANK 1"
       j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 1))
     RANK (3)
       print *, "RANK 3"
       j = INT(0, KIND=MERGE(KIND(0), -1, RANK(x) == 3))
    END SELECT
    SELECT RANK(ptr => x )
     RANK(1)
       print *, "RANK 1"
       j = INT(0, KIND=MERGE(KIND(0), -1, RANK(ptr) == 1))
     RANK (3)
       print *, "RANK 3"
       j = INT(0, KIND=MERGE(KIND(0), -1, RANK(ptr) == 3))
    END SELECT

   end subroutine

end program
