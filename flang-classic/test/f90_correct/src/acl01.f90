!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module int_getter

implicit none

contains

  subroutine foo_int(a)
    integer, intent(out) :: a
    a = 20
  end subroutine foo_int

end module int_getter

program test_function_constructor

use int_getter, only: foo_int

implicit none

abstract interface
   subroutine get_int(a)
     integer, intent(out) :: a
   end subroutine get_int
end interface

type :: foo
   procedure(get_int), nopass, pointer :: get => null()
end type foo

type(foo) :: bar
integer :: x

! This line is valid code, but is rejected.
bar = foo(foo_int)
bar%get => foo_int

call bar%get(x)

if (x .eq. 20) then
    print *, "PASS"
else
    print *, "FAIL"
end if

end program test_function_constructor
