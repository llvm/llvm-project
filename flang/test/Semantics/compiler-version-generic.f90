! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/mods.f90 2>&1 | count 0
! RUN: %flang_fc1 -fdebug-unparse -pedantic -I%t -J%t %t/test.f90 2>&1 | FileCheck %s --implicit-check-not=error --implicit-check-not=warning
! RUN: %flang_fc1 -fsyntax-only -pedantic -I%t -J%t %t/reexport.f90 2>&1 | count 0
! RUN: %flang_fc1 -fdebug-unparse -pedantic -I%t -J%t %t/reexport-use.f90 2>&1 | FileCheck %s --check-prefix=REEXPORT --implicit-check-not=error --implicit-check-not=warning

!--- mods.f90
module user_cv_co
  interface compiler_version
    module procedure okp
  end interface
  interface compiler_options
    module procedure oko
  end interface
  interface cv
    module procedure okp
  end interface
  interface uv
    module procedure okp
  end interface
  interface ieee_is_nan
    module procedure my_nan
  end interface
  type :: t
  end type
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
  logical function my_nan(x)
    type(t), intent(in) :: x
    my_nan = .false.
  end function
end module

!--- test.f90
program compiler_version_generic_use
  use user_cv_co, only: t
  call same_name_intrinsic_first()
  call same_name_user_first()
  call renamed_intrinsic_first()
  call renamed_intrinsic_user_first()
  call renamed_user_first()
  call renamed_user_intrinsic_first()
  call other_intrinsic_module(t(), 1.0)
contains
  subroutine same_name_intrinsic_first()
    use iso_fortran_env, only: compiler_version, compiler_options
    use user_cv_co, only: compiler_version, compiler_options
    ! CHECK: PRINT *, .true._4, okp(.true._4)
    ! CHECK: PRINT *, .true._4, oko(.false._4)
    print *, len(compiler_version()) > 0, compiler_version(.true.)
    print *, len(compiler_options()) > 0, compiler_options(.false.)
  end subroutine

  subroutine same_name_user_first()
    use user_cv_co, only: compiler_version, compiler_options
    use iso_fortran_env, only: compiler_version, compiler_options
    ! CHECK: PRINT *, .true._4, okp(.true._4)
    ! CHECK: PRINT *, .true._4, oko(.false._4)
    print *, len(compiler_version()) > 0, compiler_version(.true.)
    print *, len(compiler_options()) > 0, compiler_options(.false.)
  end subroutine

  subroutine renamed_intrinsic_first()
    use iso_fortran_env, only: cv => compiler_version
    use user_cv_co, only: cv
    ! CHECK: PRINT *, .true._4, okp(.true._4)
    print *, len(cv()) > 0, cv(.true.)
  end subroutine

  subroutine renamed_intrinsic_user_first()
    use user_cv_co, only: cv
    use iso_fortran_env, only: cv => compiler_version
    ! CHECK: PRINT *, .true._4, okp(.false._4)
    print *, len(cv()) > 0, cv(.false.)
  end subroutine

  subroutine renamed_user_first()
    use user_cv_co, only: compiler_version => uv
    use iso_fortran_env, only: compiler_version
    ! CHECK: PRINT *, .true._4, okp(.true._4)
    print *, len(compiler_version()) > 0, compiler_version(.true.)
  end subroutine

  subroutine renamed_user_intrinsic_first()
    use iso_fortran_env, only: compiler_version
    use user_cv_co, only: compiler_version => uv
    ! CHECK: PRINT *, .true._4, okp(.false._4)
    print *, len(compiler_version()) > 0, compiler_version(.false.)
  end subroutine

  subroutine other_intrinsic_module(x, y)
    use ieee_arithmetic, only: ieee_is_nan
    use user_cv_co, only: ieee_is_nan, t
    type(t), intent(in) :: x
    real, intent(in) :: y
    ! CHECK: PRINT *, my_nan(x)
    ! CHECK: PRINT *, __builtin_ieee_is_nan(y)
    print *, ieee_is_nan(x)
    print *, ieee_is_nan(y)
  end subroutine
end program

!--- reexport.f90
module reexport
  use iso_fortran_env, only: cv => compiler_version
  use user_cv_co, only: cv
  public :: cv
end module

!--- reexport-use.f90
program p
  use reexport
  ! REEXPORT: PRINT *, .true._4, {{.*}}okp(.true._4)
  print *, len(cv()) > 0, cv(.true.)
end program
