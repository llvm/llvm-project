!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! An initialization shall not appear if object-name is a function result

function f() result(r)
  !{error "PGF90-S-0155-Function result cannot have the PARAMETER attribute - r"}
  real, parameter :: r = 5.0
end function

function g() result(s)
  !{error "PGF90-S-0155-Function result cannot have an initializer - s"}
  real :: s = 5.0
end function

function h() result(i)
  !{error "PGF90-S-0155-A derived type type-name conflicts with function result - i"}
  type :: i
  end type
end function
