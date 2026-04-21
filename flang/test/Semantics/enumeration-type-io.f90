! RUN: %python %S/test_errors.py %s %flang_fc1
! Test I/O constraints for enumeration types (F2023 7.6.2)

module enum_io_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
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
  !ERROR: Namelist group object 'd' has a direct component 'clr' of enumeration type
  write(*, nml=nml2)
end subroutine

subroutine test_namelist_valid()
  integer :: n
  namelist /nml3/ n
  write(*, nml=nml3)
end subroutine
