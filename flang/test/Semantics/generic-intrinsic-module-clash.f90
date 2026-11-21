! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -J%t %t/mods.f90
! RUN: %flang_fc1 -fsyntax-only -pedantic -I%t -J%t %t/niladic.f90 2>&1 | FileCheck %s --check-prefix=NILADIC
! RUN: %flang_fc1 -fsyntax-only -pedantic -I%t -J%t %t/mixed.f90 2>&1 | FileCheck %s --check-prefix=MIXED
! RUN: %flang_fc1 -fsyntax-only -pedantic -I%t -J%t %t/ok.f90 2>&1 | count 0
! RUN: %flang_fc1 -fsyntax-only -I%t -J%t %t/niladic.f90 2>&1 | count 0
! RUN: %flang_fc1 -fsyntax-only -I%t -J%t %t/mixed.f90 2>&1 | count 0

!--- mods.f90
module user_niladic
  interface compiler_version
    module procedure zero
  end interface
contains
  function zero()
    character(:), allocatable :: zero
    zero = 'user'
  end function
end module

module user_subroutine
  interface compiler_version
    module procedure s
  end interface
contains
  subroutine s(f)
    logical, intent(in) :: f
  end subroutine
end module

module user_distinct
  interface compiler_version
    module procedure okp
  end interface
contains
  function okp(f)
    character(:), allocatable :: okp
    logical, intent(in) :: f
    okp = 'user'
  end function
end module

!--- niladic.f90
! NILADIC: portability: Generic 'compiler_version' should not have specific procedure 'zero' whose interface is not distinguishable from intrinsic module procedure '__builtin_compiler_version' by the rules in the standard
program p
  use iso_fortran_env, only: compiler_version
  use user_niladic
end program

!--- mixed.f90
! MIXED: portability: Generic 'compiler_version' should not have both function '__builtin_compiler_version' from an intrinsic module and subroutine 's' as specific procedures
program p
  use iso_fortran_env, only: compiler_version
  use user_subroutine
end program

!--- ok.f90
program p
  use iso_fortran_env, only: compiler_version
  use user_distinct
end program
