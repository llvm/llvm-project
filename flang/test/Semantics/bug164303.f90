!RUN: %flang -fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
module foo_mod
  use, intrinsic :: iso_fortran_env
  use, intrinsic :: iso_c_binding
  implicit none

  interface new_foo
    procedure :: foo_ctor
  end interface

contains

function foo_ctor(options) result(retval)
  implicit none
  integer, intent(in) :: options
  integer :: retval

  interface
!CHECK-NOT: error:
    subroutine new_foo(f, opt) bind(c, name='new_foo')
      import
      implicit none
      integer, intent(inout) :: f
      integer(c_int), intent(in) :: opt
    end subroutine
  end interface

  call new_foo(retval, options)
end function

end module
