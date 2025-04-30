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
     procedure :: assign_to_i 
     procedure :: assign_to_r
     procedure :: addition
  end type container
  
  interface assignment(=)
     module procedure assign_to_i
     module procedure assign_to_r
  end interface
  
  interface operator(+)
     module procedure addition
  end interface
  
  interface operator(-)
     module procedure subtraction
  end interface
  
contains
  integer function extract_i(this, ii) RESULT(iii)
    class(container) :: this
    integer ii
    iii = this%i
  end function extract_i
  
  real function extract_r(this, rr) RESULT(rrr)
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
  
  subroutine assign_to_i(this, src)
    class(container),intent(inout) :: this
    integer,intent(in) :: src
    this%i = src
  end subroutine assign_to_i
  
  subroutine assign_to_r(this, src)
    class(container),intent(inout) :: this
    real,intent(in):: src
    this%r = src
  end subroutine assign_to_r
  
  type(container) function addition(first, second) RESULT(ttt)
    class(container),intent(in) :: first
    type(container),intent(in) :: second
    type(container) :: tt
    tt%i = first%i + second%i
    tt%r = first%r + second%r
    ttt = tt
  end function addition
  
  type(container) function subtraction(first, second) RESULT(ttt)
    class(container),intent(in) :: first
    type(container),intent(in) :: second
    type(container) :: tt
    tt%i = first%i - second%i
    tt%r = first%r - second%r
    ttt = tt
  end function subtraction
  
  
  
end module my_container

program prg
USE CHECK_MOD
  use my_container
  
  class(container),allocatable :: t
  class(container),allocatable :: t2
  type(container) :: t3
  integer ei
  real er
  character ec(10)
  logical rslt(8)
  logical expect(8)
  
  rslt = .false.
  expect = .true.
  
  allocate(t) 
  allocate(t2)
  call t%init(23,4.5)
  
  ei = 0
  er = 0.0
  
  er = t%xr(1.0)
  ei = t%xi(1)
  
  rslt(1) = er .eq. 4.5
  rslt(2) = ei .eq. 23
  
  t2 = ei
  t2 = er
  
  er = t2%xr(1.0)
  ei = t2%xi(1)
  
  rslt(3) = er .eq. 4.5
  rslt(4) = ei .eq. 23
  
  t3 = t2 + t
  
  rslt(5) = t3%r .eq. 9.0
  rslt(6) = t3%i .eq. 46
  
  t3 = t3 - t
  
  rslt(7) = t3%r .eq. 4.5
  rslt(8) = t3%i .eq. 23
  
  
  call check(rslt,expect,8)
  
end program prg


