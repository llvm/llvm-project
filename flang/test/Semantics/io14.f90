! RUN: %python %S/test_errors.py %s %flang_fc1
! Test polymorphic restrictions
module m
  type base
  end type
  type, extends(base) :: t
    integer n
   contains
    procedure :: fwrite
    generic :: write(formatted) => fwrite
  end type
 contains
  subroutine fwrite(x, unit, iotype, vlist, iostat, iomsg)
    class(t), intent(in) :: x
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit, *, iostat=iostat, iomsg=iomsg) '(', iotype, ':', vlist, ':', x%n, ')'
  end subroutine
  subroutine subr(x, y, z)
    class(t), intent(in) :: x
    class(base), intent(in) :: y
    class(*), intent(in) :: z
    print *, x ! ok
    !ERROR: Derived type 'base' in I/O may not be polymorphic unless using defined I/O
    print *, y
    !ERROR: I/O list item may not be unlimited polymorphic
    print *, z
  end subroutine
end

program main
  use m
  call subr(t(123),t(234),t(345))
end
