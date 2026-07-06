! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic analysis of conditional arguments (F2023 R1526-R1528)

subroutine test_conditional_arg
  implicit none
  integer :: a, b, c
  logical :: flag, flag2

  ! Simple conditional arg
  !ERROR: not yet implemented: Fortran 2023 conditional arguments are not yet supported
  call sub((flag ? a : b))

  ! Multi-branch conditional arg
  !ERROR: not yet implemented: Fortran 2023 conditional arguments are not yet supported
  call sub((flag ? a : flag2 ? b : c))

  ! .NIL. in else position
  !ERROR: not yet implemented: Fortran 2023 conditional arguments are not yet supported
  call sub((flag ? a : .NIL.))

  ! Keyword argument with conditional arg
  !ERROR: not yet implemented: Fortran 2023 conditional arguments are not yet supported
  call sub(arg = (flag ? a : b))

  ! .NIL. in both branches
  !ERROR: not yet implemented: Fortran 2023 conditional arguments are not yet supported
  call sub((flag ? .NIL. : .NIL.))

end subroutine
