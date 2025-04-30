!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program test_section_constructor

implicit none

type, abstract :: foo_base
end type foo_base

type, extends(foo_base) :: foo
   integer, allocatable :: a(:)
end type foo

type(foo) :: b

integer :: a1(2) = 0
integer :: a2(2,2) = 0

b = foo(a1)      ! OK
b = foo(a2(:,1)) ! Spurious warning

print *, "PASS"

end program test_section_constructor
