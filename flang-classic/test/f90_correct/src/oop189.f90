! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module shape_mod
  
  type shape
     integer :: color
     logical :: filled
     integer :: x
     integer :: y
  end type shape
  
  type, EXTENDS ( shape ) :: rectangle
  integer :: the_length
  integer :: the_width
end type rectangle

type, extends (rectangle) :: square
end type square

end module shape_mod

logical function test_type(t)
  use shape_mod
  class(*) :: t
  type(square) :: sq
  test_type = extends_type_of(t,sq)
end function test_type

logical function test_type_allo(t) RESULT(R)
  use shape_mod
  class(*),allocatable :: t
  type(square) :: sq
  r = extends_type_of(sq,t)
end function test_type_allo

logical function test_type_ptr(t,v) RESULT(R)
  class(*),pointer :: t
  class(*),pointer :: v 
  r = extends_type_of(t,v)
end function test_type_ptr

program unlimited_poly
USE CHECK_MOD
  
  use shape_mod
  
  interface
     
     logical function test_type(t)
       class(*) :: t
     end function test_type
     
     logical function test_type_allo(t) RESULT(R)
       class(*),allocatable :: t
     end function test_type_allo
    
     logical function test_type_ptr(t,v) RESULT(R)
       class(*),pointer :: t
       class(*),pointer :: v
     end function test_type_ptr
     
  end interface
  
  logical l 
  logical results(21)
  logical expect(21)
  type(square) :: s
  class(rectangle),allocatable,target :: r
  class(*), allocatable :: a
  class(*), pointer :: p
  class(*), pointer :: rp
  class(square), pointer :: sq
  
  results = .false.
  expect = .true.
  
  results(1) = test_type(s)
  results(2) = extends_type_of(s,a)
  results(3) = extends_type_of(a,s) .eq. .false.
  results(4) = same_type_as(s,a) .eq. .false.
  results(5) = same_type_as(a,s) .eq. .false.
  allocate(rectangle::a)
  results(6) = test_type_allo(a)
  results(7) = same_type_as(a,r)
  
  nullify(p)
  results(8) = extends_type_of(a,p)
  results(9) = extends_type_of(p,a) .eq. .false.
  results(10) = same_type_as(a,p) .eq. .false.
  results(11) = same_type_as(p,a) .eq. .false.
  
  allo : select type(a)
  type is(rectangle) allo
     a%the_length = 99
     a%the_width = 11
     shape : associate(sh=>a%shape)
       sh%color = 66
       sh%filled = .true.
       sh%x = 8
       sh%y = 7
     end associate shape
  end select allo


  select type(a)
  class is (rectangle)
    allocate(r,source=a)
  end select

  results(12) = r%the_length .eq. 99
  results(13) = r%the_width .eq. 11

  results(14) = r%color .eq. 66
  results(15) = r%filled
  results(16) = r%x .eq. 8
  results(17) = r%y .eq. 7
 
  rp => r 
  results(18) = test_type_ptr(rp,p)

  results(19) = test_type_ptr(p,rp) .eq. .false.

  p => r
  nullify(sq)
  results(20) = test_type_ptr(p,sq)

  allocate(sq)
  results(21) = test_type_ptr(p,sq) .eq. .false.
 
  call check(results,expect,21)
  
end program unlimited_poly


