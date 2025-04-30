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
     procedure :: assign_to_i 
     procedure :: assign_to_r
     procedure,pass(second) :: addition => addit
     procedure,pass(second) :: addition_array
     generic :: assignment(=) => assign_to_i, assign_to_r
     generic :: operator(+) => addition
     generic :: operator(+) => addition_array
     procedure,pass(second) :: add_array
  end type container

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

  type(container) function addit(first, second) RESULT(ttt)
  class(container),intent(in) :: second
  type(container),intent(in) :: first
  type(container) :: tt
  tt%i = first%i + second%i
  tt%r = first%r + second%r
  ttt = tt

  tt = second

  end function addit

  type(container) function addition_array(first, second) RESULT(ttt)
  class(container),intent(in) :: second
  type(container),intent(in) :: first(:)
  type(container) :: tt
  integer sz

  sz = size(first)
  do i=1, sz
    tt%i = first(i)%i + second%i
    tt%r = first(i)%r + second%r
  enddo
  ttt = tt
  end function addition_array

  type(container) function add_array(first, second) RESULT(ttt)
  class(container),intent(in) :: second
  type(container),intent(in) :: first(:)

  ttt = first + second

  end function add_array




end module my_container

program prg
USE CHECK_MOD
  use my_container


  class(container),allocatable :: t
  class(container),allocatable :: t2
  type(container) :: t3
  type(container) :: t_array(10)
  integer ei
  real er
  character ec(10)
  logical rslt(6)
  logical expect(6)
  integer i
  real r
  
  rslt = .false.
  expect = .true.
  
  allocate(t) 
  allocate(t2)
  call t%init(23,4.5)

  ei = 0
  er = 0.0

  er = t%extract(1.0)
  ei = t%extract(1)

  rslt(1) = er .eq. 4.5
  rslt(2) = ei .eq. 23

  t2 = ei
  t2 = er

  er = t2%extract(1.0)
  ei = t2%extract(1)

  rslt(3) = er .eq. 4.5
  rslt(4) = ei .eq. 23

  do i=1,10
    r = i
    call t_array(i)%init(i,r)
  enddo

  call t%init(0,0.0)
  !t3 = t_array + t

  t3 = t%add_array(t_array)

  rslt(5) = t3%extract(1.0) .eq. 10.0
  rslt(6) = t3%extract(1) .eq. 10

  call check(rslt,expect,6)
  
end program prg


