! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -pedantic -J%t %t/mods.f90 2>&1 | count 0
! RUN: %flang_fc1 -fdebug-unparse -pedantic -I%t -J%t %t/user-first.f90 2>&1 | FileCheck %s --check-prefix=USER-FIRST --implicit-check-not=error --implicit-check-not=warning
! RUN: %flang_fc1 -fdebug-unparse -pedantic -I%t -J%t %t/intrinsic-first.f90 2>&1 | FileCheck %s --check-prefix=INTRINSIC-FIRST --implicit-check-not=error --implicit-check-not=warning
! RUN: %flang_fc1 -fdebug-unparse -pedantic -I%t -J%t %t/homonym.f90 2>&1 | FileCheck %s --check-prefix=HOMONYM --implicit-check-not=error --implicit-check-not=warning

!--- mods.f90
module has_sum
  intrinsic :: sum, sin
  public :: sum, sin
end module

module user_sum
  interface sum
    module procedure sum
  end interface
  interface sin
    module procedure sin
  end interface
contains
  function sum(x, y)
    real, intent(in) :: x, y
    real :: sum
    sum = -99.0
  end function
  function sin(x, y)
    real, intent(in) :: x, y
    real :: sin
    sin = x + y
  end function
end module

!--- user-first.f90
program p
  use user_sum
  use has_sum, only: sum, sin
  ! USER-FIRST: PRINT *, 3._4
  ! USER-FIRST: PRINT *, sum(1._4,2._4)
  ! USER-FIRST: PRINT *, 0._4
  ! USER-FIRST: PRINT *, sin(1._4,2._4)
  print *, sum([1.0, 2.0])
  print *, sum(1.0, 2.0)
  print *, sin(0.0)
  print *, sin(1.0, 2.0)
  print *, apply(sum)
contains
  function apply(fn)
    interface
      function fn(x, y)
        real, intent(in) :: x, y
        real :: fn
      end function
    end interface
    real :: apply
    apply = fn(1.0, 2.0)
  end function
end program

!--- intrinsic-first.f90
program p
  use has_sum, only: sum, sin
  use user_sum
  ! INTRINSIC-FIRST: PRINT *, 3._4
  ! INTRINSIC-FIRST: PRINT *, sum(1._4,2._4)
  ! INTRINSIC-FIRST: PRINT *, 0._4
  ! INTRINSIC-FIRST: PRINT *, sin(1._4,2._4)
  print *, sum([1.0, 2.0])
  print *, sum(1.0, 2.0)
  print *, sin(0.0)
  print *, sin(1.0, 2.0)
end program

!--- homonym.f90
module homonym_cv
  interface compiler_version
    module procedure compiler_version
  end interface
contains
  function compiler_version(f)
    character(:), allocatable :: compiler_version
    logical, intent(in) :: f
    compiler_version = 'user'
  end function
end module

program p
  use iso_fortran_env, only: compiler_version
  use homonym_cv
  ! HOMONYM: PRINT *, .true._4
  ! HOMONYM: PRINT *, compiler_version(.true._4)
  print *, len(compiler_version()) > 0
  print *, compiler_version(.true.)
end program
