! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for I/O of derived types without defined I/O procedures
! but with exposed allocatable/pointer components that would fail
! at run time.

module m1
  type :: poison
    real, allocatable :: allocatableComponent(:)
  end type
  type :: ok
    integer :: x
    type(poison) :: pill
   contains
    procedure :: wuf1
    generic :: write(unformatted) => wuf1
  end type
  type :: maybeBad
    integer :: x
    type(poison) :: pill
  end type
 contains
  subroutine wuf1(dtv, unit, iostat, iomsg)
    class(ok), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit) dtv%x
  end subroutine
end module

module m2
  use m1
  interface write(unformatted)
    module procedure wuf2
  end interface
 contains
  subroutine wuf2(dtv, unit, iostat, iomsg)
    class(maybeBad), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit) dtv%x
  end subroutine
end module

module m3
  use m1
 contains
  subroutine test3(u)
    integer, intent(in) :: u
    type(ok) :: x
    type(maybeBad) :: y
    type(poison) :: z
    write(u) x ! always ok
    !ERROR: Derived type 'maybebad' in I/O cannot have an allocatable or pointer direct component 'allocatablecomponent' unless using defined I/O
    write(u) y ! bad here
    !ERROR: Derived type 'poison' in I/O cannot have an allocatable or pointer direct component 'allocatablecomponent' unless using defined I/O
    write(u) z ! bad
  end subroutine
end module

module m4
  use m2
 contains
  subroutine test4(u)
    integer, intent(in) :: u
    type(ok) :: x
    type(maybeBad) :: y
    type(poison) :: z
    write(u) x ! always ok
    write(u) y ! ok here
    !ERROR: Derived type 'poison' in I/O cannot have an allocatable or pointer direct component 'allocatablecomponent' unless using defined I/O
    write(u) z ! bad
  end subroutine
end module

