! RUN: %flang_fc1 -fsyntax-only %s
! Test merging USE of ISO_FORTRAN_ENV compiler_version/compiler_options
! with a user generic of the same local name.

module user_cv_co
  interface compiler_version
    module procedure okp
  end interface
  interface compiler_options
    module procedure oko
  end interface
contains
  function okp(f)
    character(:), allocatable :: okp
    logical, intent(in) :: f
    okp = merge('user:ok  ', 'user:fail', f)
  end function
  function oko(f)
    character(:), allocatable :: oko
    logical, intent(in) :: f
    oko = merge('opts:ok  ', 'opts:fail', f)
  end function
end module

program compiler_version_generic_use
  call order1()
  call order2()
contains
  subroutine order1()
    use iso_fortran_env, only: compiler_version, compiler_options
    use user_cv_co
    character(*), parameter :: bufv = compiler_version()
    character(*), parameter :: bufo = compiler_options()
    print *, bufv
    print *, bufo
    print *, compiler_version(.true.), ' or ', compiler_version(.false.)
    print *, compiler_options(.true.), ' or ', compiler_options(.false.)
  end subroutine

  subroutine order2()
    use user_cv_co
    use iso_fortran_env, only: compiler_version, compiler_options
    character(*), parameter :: bufv = compiler_version()
    character(*), parameter :: bufo = compiler_options()
    print *, bufv
    print *, bufo
    print *, compiler_version(.true.), ' or ', compiler_version(.false.)
    print *, compiler_options(.true.), ' or ', compiler_options(.false.)
  end subroutine
end program
