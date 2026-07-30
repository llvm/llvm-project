! RUN: %python %S/test_errors.py %s %flang_fc1 -fenumeration-type
! Test I/O constraints for enumeration types (F2023 7.6.2)

module enum_io_mod
  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  ! Wraps an enumeration component behind defined output I/O so it can be
  ! used to test that a component with defined I/O shields whatever is
  ! nested inside it from the enumeration-type component check.
  type :: has_color_io
    type(color) :: c
  contains
    procedure :: wfc
    generic :: write(formatted) => wfc
  end type
contains
  subroutine wfc(x, unit, iotype, vlist, iostat, iomsg)
    class(has_color_io), intent(in) :: x
    integer, intent(in) :: unit
    character(*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit, '(I4)', iostat=iostat, iomsg=iomsg) x%c
  end subroutine
end module

subroutine test_valid_io()
  use enum_io_mod
  type(color) :: c
  character(10) :: fmt
  c = red
  fmt = '(I4)'
  ! Valid: explicit format with I edit descriptor
  write(*, '(I4)') c
  ! Valid: explicit format via character variable
  write(10, fmt) c
  ! Valid: explicit format read
  read(*, '(I4)') c
end subroutine

subroutine test_list_directed()
  use enum_io_mod
  type(color) :: c
  c = red
  !ERROR: Enumeration type may not appear in list-directed output
  print *, c
  !ERROR: Enumeration type may not appear in list-directed input
  read *, c
end subroutine

subroutine test_unformatted()
  use enum_io_mod
  type(color) :: c
  c = red
  !ERROR: Enumeration type may not be used in unformatted I/O
  write(10) c
  !ERROR: Enumeration type may not be used in unformatted I/O
  read(10) c
end subroutine

subroutine test_component_io()
  use enum_io_mod
  type :: has_color
    type(color) :: c
  end type
  type(has_color) :: d
  !ERROR: List-directed output item has a component 'c' of enumeration type
  print *, d
  !ERROR: List-directed input item has a component 'c' of enumeration type
  read *, d
  !ERROR: Enumeration type may not be used in unformatted I/O
  write(10) d
  !ERROR: Enumeration type may not be used in unformatted I/O
  read(10) d
end subroutine

subroutine test_shielded_component()
  ! A component whose type has defined output I/O is treated as a single
  ! effective item (F2023 12.6.3) and is not expanded, so the enumeration
  ! type nested inside has_color_io is shielded from the list-directed
  ! output check.  This is expected to compile without error.
  use enum_io_mod
  type :: wrapper
    type(has_color_io) :: hc
  end type
  type(wrapper) :: w
  w%hc%c = red
  print *, w
end subroutine

subroutine test_nested_rejection()
  ! The enumeration type is nested two levels deep through a plain
  ! intermediate type with no defined I/O, so the recursive component
  ! search must still find and reject it.
  use enum_io_mod
  type :: inner
    type(color) :: c
  end type
  type :: outer
    type(inner) :: i
  end type
  type(outer) :: o
  o%i%c = red
  !ERROR: List-directed output item has a component 'c' of enumeration type
  print *, o
  !ERROR: List-directed input item has a component 'c' of enumeration type
  read *, o
end subroutine

subroutine test_namelist_enum_object()
  use enum_io_mod
  type(color) :: c
  namelist /nml/ c
  !ERROR: Enumeration type 'color' may not be a namelist group object
  write(*, nml=nml)
end subroutine

subroutine test_namelist_enum_component()
  use enum_io_mod
  type :: has_color
    type(color) :: clr
    integer :: n
  end type
  type(has_color) :: d
  namelist /nml2/ d
  !ERROR: Namelist group object 'd' has a component 'clr' of enumeration type
  write(*, nml=nml2)
end subroutine

subroutine test_namelist_valid()
  integer :: n
  namelist /nml3/ n
  write(*, nml=nml3)
end subroutine
