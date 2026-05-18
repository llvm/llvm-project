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

! Regression test: an illegal recursive derived-type component used to cause
! infinite recursion in FindUnsafeIoDirectComponent when the object appeared
! in an I/O list (issue #192387).
subroutine test_recursive_io
  type t1
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(t1) :: b
  end type t1
  type(t1) :: obj
  print *, obj
end subroutine

! Same regression covering the FindInaccessibleComponent walk: the type
! must be defined in a module and used in I/O outside that module so the
! recursive component traversal in FindInaccessibleComponent is reached.
module m_recursive
  type t2
    !ERROR: Recursive use of the derived type requires POINTER or ALLOCATABLE
    type(t2) :: b
  end type t2
end module
subroutine test_recursive_io_module
  use m_recursive
  type(t2) :: obj
  print *, obj
end subroutine

! Positive cases: a recursive type is legal when the recursive component
! is POINTER or ALLOCATABLE.  With defined I/O, an I/O list item of such
! a type is accepted without diagnostics.
module m_recursive_pointer
  type :: rp
    integer :: x
    type(rp), pointer :: next => null()
   contains
    procedure :: wuf_rp
    generic :: write(unformatted) => wuf_rp
  end type
 contains
  subroutine wuf_rp(dtv, unit, iostat, iomsg)
    class(rp), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit) dtv%x
  end subroutine
end module
subroutine test_recursive_pointer_io(u)
  use m_recursive_pointer
  integer, intent(in) :: u
  type(rp) :: obj
  write(u) obj ! ok: defined I/O
end subroutine

module m_recursive_allocatable
  type :: ra
    integer :: x
    type(ra), allocatable :: next
   contains
    procedure :: wuf_ra
    generic :: write(unformatted) => wuf_ra
  end type
 contains
  subroutine wuf_ra(dtv, unit, iostat, iomsg)
    class(ra), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit) dtv%x
  end subroutine
end module
subroutine test_recursive_allocatable_io(u)
  use m_recursive_allocatable
  integer, intent(in) :: u
  type(ra) :: obj
  write(u) obj ! ok: defined I/O
end subroutine

