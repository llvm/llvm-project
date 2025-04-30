!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for RESULT in entry and function

module m
  implicit none
contains
  function func(a) result(res)
    implicit none
    integer :: a, res
    real :: func2
    !{error "PGF90-S-0072-Assignment operation illegal to func - should use res"}
    func = a*2
    return
  entry ent(a) result(func2)
    !{error "PGF90-S-0072-Assignment operation illegal to ent - should use func2"}
    ent = -a*0.5
    return
  end function func
end module m
