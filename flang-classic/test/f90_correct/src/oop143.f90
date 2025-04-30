! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       


module my_container
  
  type container
     integer i
     real r
   contains
     procedure :: init => init_container
     procedure :: xi => extract_i
     procedure :: xr => extract_r
     generic :: extract => xi, xr
  end type container

contains
  integer function extract_i(this,ii) RESULT(iii)
    class(container) :: this
    integer ii
    iii = this%i
  end function extract_i
  
  real function extract_r(this,rr) RESULT(rrr)
    class(container) :: this
    real rr
    rrr = this%r
  end function extract_r
  
  subroutine init_container(this, ic, ir)
    class(container) :: this
    integer :: ic
    real :: ir
    this%i = ic
    this%r = ir
  end subroutine init_container
  
  
end module my_container

program prg
USE CHECK_MOD
  use my_container

  type, extends(container) :: container2
  end type container2

  class(container2),allocatable :: t
  integer ei
  real er
  logical rslt(2)
  logical expect(2)
  
  rslt = .false.
  expect = .true.
  
  allocate(t) 
  call t%init(23,4.5)
  
  ei = 0
  er = 0.0
  er = t%extract(4.5)
  ei = t%extract(23)

  rslt(1) = er .eq. 4.5
  rslt(2) = ei .eq. 23
  
  call check(rslt,expect,2)
  
end program prg


