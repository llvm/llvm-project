!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Simply check the file compiles with no error.
!
! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s
! CHECK: version_var
! CHECK: options_var

module usermod
  use iso_fortran_env
end module usermod

program cp2k
  use usermod
  use iso_fortran_env, only: compiler_version, compiler_options
  implicit none
  character(len=:), allocatable :: version_var
  character(len=:), allocatable :: options_var
  version_var = compiler_version()
  options_var = compiler_options()
  print *, version_var
  print *, options_var
end program cp2k
