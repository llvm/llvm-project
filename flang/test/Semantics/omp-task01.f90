! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! 2.9.1 task Construct
! Invalid entry to OpenMP structured block.

recursive subroutine traverse ( P )
  type Node
    type(Node), pointer :: left, right
  end type Node

  type(Node) :: P

  !CHECK: invalid branch into an OpenMP structured block
  goto 10

  if (associated(P%left)) then
    !$omp task
    call traverse(P%left)
    !CHECK: In the enclosing TASK directive branched into
    !CHECK: STOP statement is not allowed in a TASK construct
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
