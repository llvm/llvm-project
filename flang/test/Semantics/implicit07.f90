! RUN: %python %S/test_errors.py %s %flang_fc1
implicit none(external)
external x
integer :: f, i, arr(1) = [0]
call x
!ERROR: 'y' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
call y
!ERROR: 'f' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
i = f()
block
  !ERROR: 'z' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  call z
end block
print *, arr(1) ! no error
end

! Function dummy argument without external
subroutine sub(func_arg, n)
  implicit none(external)
  integer :: func_arg
  integer :: n
  !ERROR: 'func_arg' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  n = func_arg(n)
end subroutine

! Subroutine dummy arguments with/without external specified
subroutine sub_call(p, q)
  implicit none(external)
  ! error without external, no error with
  external p
  call p
  !ERROR: 'q' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  call q
end subroutine

! Case that works because EXTERNAL is explicitly specified
subroutine sub_works(func_arg, n)
  implicit none(external)
  integer, external :: func_arg
  integer :: n
  n = func_arg(n)
end subroutine

! Case that works because interface is specified
subroutine sub_interface(func_arg, n)
  implicit none(external)
  interface
    function func_arg(x)
      integer, intent(in) :: x
      integer :: func_arg
    end function
  end interface
  integer :: n
  n = func_arg(n)
end subroutine

! Inherited implicit none(external) into contains routine.
subroutine sub_host(func_arg, n)
  implicit none(external)
  integer, external :: func_arg
  integer :: n
  n = func_arg(n)
contains
  subroutine inner(g)
    integer :: g
    !ERROR: 'g' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
    n = g(n)
  end subroutine
end subroutine

! Tests module scope for implicit none(external)
module m
  implicit none(external)
contains
  subroutine mod_sub(func_arg, n)
    integer :: func_arg
    integer :: n
    !ERROR: 'func_arg' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
    n = func_arg(n)
  end subroutine
end module
