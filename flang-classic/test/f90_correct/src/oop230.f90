! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module shape_mod
type :: shape
        integer :: color
        logical :: filled
        integer :: x
        integer :: y
contains
	procedure :: write => write_shape 
	procedure :: draw => draw_shape
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
contains
        procedure,pass(this) :: write => write_rec
	procedure :: draw => draw_rectangle
end type rectangle

type, extends (rectangle) :: square
contains
        procedure :: draw => draw_sq
	procedure :: write => write_sq
	procedure,pass(this) :: write2 => write_sq2
	procedure,pass(this) :: write3 => write_sq3

end type square
contains

  subroutine write_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   type(shape) :: sh
   results(i) = .true.!same_type_as(sh,this)
   end subroutine write_shape

   subroutine write_rec(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(shape) :: sh
   results(i) = .true.!same_type_as(sh,this)
   end subroutine write_rec

   subroutine draw_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   print *, 'draw shape!'
   end subroutine draw_shape

   subroutine draw_rectangle(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   results(i) = .true.!extends_type_of(this,rec)
   end subroutine draw_rectangle

   subroutine write_sq(this,results,i)
   class (square) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   results(i) = .true.!extends_type_of(this,rec)
   end subroutine write_sq

   subroutine draw_sq(this,results,i)
   class (square) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   results(i) = .true.!extends_type_of(this,rec)
   end subroutine draw_sq

   subroutine write_sq2(i,this,results)
   class (square) :: this
   integer i 
   logical results(:)
   type(rectangle) :: rec
   results(i) = .true.!extends_type_of(this,rec)
   end subroutine write_sq2

   logical function write_sq3(o,this)
   integer, optional :: o
   class (square) :: this
   type(rectangle) :: rec
   write_sq3 = .true.!extends_type_of(this,rec)
   end function 

end module shape_mod

program p
USE CHECK_MOD
use shape_mod

logical l 
logical results(5)
logical expect(5)
class(square),allocatable :: s
class(shape),allocatable :: sh
type(rectangle) :: r
type(square),allocatable :: sq

expect = .true.
results = .false.

allocate(s)
call s%write2(1,results)
call s%write(results,2)
call s%draw(results,3)

allocate(sq)
results(5) = sq%write3(1)

allocate(sh)
call write_shape(sh,results,4)

call check(results,expect,5)

end


