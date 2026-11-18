! RUN: %python %S/test_errors.py %s %flang_fc1

! A module that re-exports an ambiguous (never-referenced, and therefore
! legal per F2023 14.2.2 p8) USE-associated name must still diagnose a
! reference to that name as ambiguous, including through a renaming
! only-list USE, within a single compilation (as opposed to across
! separately compiled module files, where the ambiguous name is omitted
! from the producing module's own module file; see modfile89.f90).
module pk133
  integer :: x
end module
module vm133
  integer :: x
end module
module middle133
  use pk133
  use vm133
end module
module down133
  use middle133
contains
  subroutine check
    ! ERROR: Reference to 'x' is ambiguous
    print *, x
  end subroutine
end module
module down133_rename
  use middle133, only: y => x
contains
  subroutine check
    ! ERROR: Reference to 'y' is ambiguous
    print *, y
  end subroutine
end module
