! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests "generic procedure interfaces named the same as a type in the same 
! module", a.k.a. "Type Overloading"


! This is a slightly modified version of Figure 17.1 in
! Fortran 95/2003 Explained by Metcalf, Reid & Cohen.
! All modifications are marked.
!

module mycomplex_module        ! Not in Figure 17.1
    private                     ! Not in Figure 17.1
    public :: mycomplex         ! Not in Figure 17.1
    type mycomplex
       private                  ! Not in Figure 17.1
       real :: argument, modulus
       real :: x, y
       contains
       procedure :: getx
       procedure :: gety
    end type
    interface mycomplex
       module procedure complex_to_mycomplex, two_reals_to_mycomplex
    end interface
    contains
       type(mycomplex) function complex_to_mycomplex(c)
          complex, intent(in) :: c
	  complex_to_mycomplex%x = 0.0
	  complex_to_mycomplex%y = 0.0
       end function complex_to_mycomplex
       type(mycomplex) function two_reals_to_mycomplex(x,y)
          real, intent(in)           :: x
          real, intent(in), optional :: y
	  two_reals_to_mycomplex%x = x
	  if (present(y)) then
	     two_reals_to_mycomplex%y = y
	  else
	     two_reals_to_mycomplex%y = 0.0
	  endif
        end function two_reals_to_mycomplex
       real function getx(this)
        class(mycomplex) :: this
	getx = this%x
       end function getx
       real function gety(this)
	class(mycomplex) :: this
	gety = this%y
       end function gety
end module mycomplex_module    ! Not in Figure 17.1

program myuse                  ! Not in Figure 17.1
     use mycomplex_module       ! Not in Figure 17.1
     logical rslt(4), expect(4)
     type(mycomplex) :: c       ! Not in Figure 17.1
     rslt = .false.
     expect = .true.
     c = mycomplex(x=1.0, y=2.0)! This should invoke two_reals_tomycomplex()
     rslt(1) = c%getx() .eq. 1.0
     rslt(2) = c%gety() .eq. 2.0
     c = mycomplex(3.0, 4.0)    ! This should invoke two_reals_tomycomplex()
     rslt(3) = c%getx() .eq. 3.0
     rslt(4) = c%gety() .eq. 4.0

     call check(rslt,expect,4)

end program myuse

