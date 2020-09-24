! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.9.1 task Construct
! Invalid entry to OpenMP structured block.

recursive subroutine traverse ( P )
  type Node
    type(Node), pointer :: left, right
  end type Node

  type(Node) :: P

  !ERROR: invalid entry to OpenMP structured block
  goto 10

  if (associated(P%left)) then
    !$omp task
    call traverse(P%left)
    10 stop
    !$omp end task
  endif

  if (associated(P%right)) then
    !$omp task
    call traverse(P%right)
    !$omp end task
    endif
  call process ( P )

 end subroutine traverse
