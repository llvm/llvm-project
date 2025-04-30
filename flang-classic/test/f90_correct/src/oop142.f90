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
     procedure,nopass :: xi => extract_i
     procedure,nopass :: xr => extract_r
     generic :: extract => xi, xr
  end type container
  
  type, extends(container) :: container2
  character c
contains
  procedure :: init => init_container2
  procedure,nopass :: xc => extract_c
  generic :: extract => xc
end type container2


contains
  subroutine extract_i(ii)
    integer ii
    ii = 23
  end subroutine extract_i
  
  subroutine extract_r(rr)
    real rr
    rr = 4.5
  end subroutine extract_r
  
  subroutine extract_c(cc)
    character cc
    cc = 'Z'
  end subroutine extract_c
  
  subroutine init_container2(this, ic, ir, c)
    class(container2) :: this
    integer :: ic
    real :: ir
    character,optional :: c
    this%c = c
    call this%container%init(ic, ir)
  end subroutine init_container2
  
  subroutine init_container(this, ic, ir, c)
    class(container) :: this
    integer :: ic
    real :: ir
    character,optional  :: c
    this%i = ic
    this%r = ir
  end subroutine init_container
  
  
end module my_container

program prg
USE CHECK_MOD
  use my_container
  
  class(container2),allocatable :: t
  integer ei
  real er
  character ec
  logical rslt(3)
  logical expect(3)
  
  rslt = .false.
  expect = .true.
  
  allocate(t) 
  call t%init(23,4.5,'Z')
  
  
  ei = 0
  er = 0.0
  ec = 'A'
  
  call t%extract(er)
  call t%extract(ei)
  call t%extract(ec)
  
  rslt(1) = er .eq. 4.5
  rslt(2) = ei .eq. 23
  rslt(3) = ec .eq. 'Z'
  
  call check(rslt,expect,3)
  
end program prg


