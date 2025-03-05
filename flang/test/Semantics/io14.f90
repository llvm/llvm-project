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
  type, extends(t) :: t2
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
  subroutine subr(x, y, z, w)
    class(t), intent(in) :: x
    class(base), intent(in) :: y
    class(*), intent(in) :: z
    class(t2), intent(in) :: w
    print *, x ! ok
    print *, w ! ok
    !ERROR: Derived type 'base' in I/O may not be polymorphic unless using defined I/O
    print *, y
    !ERROR: I/O list item may not be unlimited polymorphic
    print *, z
  end subroutine
end
