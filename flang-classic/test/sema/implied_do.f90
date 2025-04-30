!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang1 %s | FileCheck %s
! CHECK: procedure:Program
! CHECK-DAG: s:[[FUNC:[0-9]+]] {{.*}}:func
! CHECK-DAG: s:[[REALLOC:[0-9]+]] {{.*}}:f90_realloc_arr_in_impiled_do
! CHECK-DAG: s:[[DEALLOC:[0-9]+]] {{.*}}:f90_dealloc03a
! CHECK-COM: begin of implied-do loop
! CHECK: DOBEG
! CHECK-NOT: DOEND
! CHECK-COM: call func() in the loop and realloc temp array before copying
! CHECK: UCALL n{{[0-9]+}} s[[FUNC]]
! CHECK: UCALL n6 s[[REALLOC]]
! CHECK-COM: loop for copying from return variable
! CHECK: DOBEG
! CHECK: DOEND
! CHECK-COM: dealloc return variable in the implied-do loop
! CHECK: CALL n4 s[[DEALLOC]]
! CHECK-COM: end of implied-do loop
! CHECK: DOEND
! CHECK-COM: loop for assignment to x
! CHECK: DOBEG
! CHECK: DOEND
! CHECK-COM: dealloc temp array outside the implied-do loop
! CHECK: CALL n4 s[[DEALLOC]]
! CHECK-NOT: CALL n4 s[[DEALLOC]]
! CHECK: procedure:Subroutine

program main
  implicit none
  integer :: x(2) = 0
  integer :: i

  x = (/ (func(1), i = 1, 2) /)
  call check()
contains
  function func(n)
    integer :: n
    integer :: func(n)
    func = n
  end function
  subroutine check()
    if (any(x /= (/1, 1/))) STOP 1
  end subroutine
end program
