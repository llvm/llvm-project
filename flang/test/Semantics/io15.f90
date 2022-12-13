! RUN: %python %S/test_errors.py %s %flang_fc1
! Test visibility restrictions
module m
  type t1
    integer, private :: ip1 = 123
   contains
    procedure :: fwrite1
    generic :: write(formatted) => fwrite1
  end type t1
  type t2
    integer, private :: ip2 = 234
    type(t1) x1
  end type t2
  type t3
    type(t1) x1
    type(t2) x2
  end type t3
  type, extends(t2) :: t4
  end type t4
 contains
  subroutine fwrite1(x, unit, iotype, vlist, iostat, iomsg)
    class(t1), intent(in) :: x
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit, *, iostat=iostat, iomsg=iomsg) '(', iotype, ':', vlist, ':', x%ip1, ')'
  end subroutine
  subroutine local ! all OK since type is local
    type(t1) :: x1
    type(t2) :: x2
    type(t3) :: x3
    type(t4) :: x4
    print *, x1
    print *, x2
    print *, x3
    print *, x4
  end subroutine
end module

program main
  use m
  type(t1) :: x1
  type(t2) :: x2
  type(t3) :: x3
  type(t4) :: x4
  print *, x1 ! ok
  !ERROR: I/O of the derived type 't2' may not be performed without defined I/O in a scope in which a direct component like 'ip2' is inaccessible
  print *, x2
  !ERROR: I/O of the derived type 't3' may not be performed without defined I/O in a scope in which a direct component like 'ip2' is inaccessible
  print *, x3
  !ERROR: I/O of the derived type 't4' may not be performed without defined I/O in a scope in which a direct component like 'ip2' is inaccessible
  print *, x4
end
