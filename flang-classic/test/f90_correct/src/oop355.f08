! Part of the LLVM Project, under the Apache License v2.0 with LLVM          Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests "generic procedure interfaces named the same as a type in the same
! module", a.k.a. "Type Overloading"

! This is a slightly modified version of Figure 17.1 in
! Fortran 95/2003 Explained by Metcalf, Reid & Cohen.
! All modifications are marked.
!

!this module is check for accessing member variables of type real16

module oop_quad_module
  private
  public :: myquad
  type myquad
    real(16) :: x
    contains
      procedure :: getx
  end type
    contains
      real(16) function getx(this)
        class(myquad) :: this
        getx = this%x
      end function getx
end module oop_quad_module

program myuse
  use oop_quad_module
    logical rslt, expect
    type(myquad) :: a
    rslt = .false.
    expect = .true.

    a = myquad(x = 1.0)
    rslt = a%getx() .eq. 1.0

    call check(rslt,expect,1)

end program myuse
