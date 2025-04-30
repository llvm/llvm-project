!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Check that the type order between the declaration of "pure function" and its implementation
! doesnt produce a compilation error
! Test related to fix for issue #1016

module pureFunction
  implicit none
  private
  public :: func, other_func
  type :: foo
    integer :: mydim
  end type

contains

subroutine func(tparam, oparam)
  integer, intent(in) :: tparam
  integer, dimension(other_func(tparam, 10)), intent(out), optional :: oparam

  oparam = 42
end subroutine

pure function other_func(tparam, idim)
  integer, intent(in) :: tparam
  integer, intent(in) :: idim
  integer :: other_func

  other_func = min(tparam, idim)
end function

end module

program testPureFunction
	use pureFunction
	implicit none

	integer :: result(1)
	integer :: expect
	integer :: param1
  

	param1 = 20	
	call func(param1,result)
	expect = 42

	call check(result, expect, 1)
	print *,result, expect, param1

end program
