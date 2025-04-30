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
contains
        procedure,pass(this) :: write => write_shape
        procedure :: draw => draw_shape
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
contains
        procedure,pass(this) :: write => write_rec
        procedure :: draw => draw_rectangle
	procedure :: draw2
	procedure,pass(this) :: write2 => write_rec2
end type rectangle

type, extends (rectangle) :: square
contains
        !procedure :: draw => draw_sq
        !procedure,pass(this) :: write => write_sq
        procedure,pass(this) :: write2 => write_sq2

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
   type(rectangle) :: rec
   results(i) = .true.!same_type_as(rec,this)
   end subroutine write_rec

   subroutine write_rec2(results,i,o,this)
   class (rectangle) :: this
   logical results(:)
   integer i
   integer, optional :: o
   type(rectangle) :: rec
   results(i) = .true.!same_type_as(rec,this)
   end subroutine write_rec2

   subroutine draw_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   end subroutine draw_shape

   subroutine draw_rectangle(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   results(i) = .true.!same_type_as(this,rec)
   end subroutine draw_rectangle

   subroutine draw2(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   results(i) = .true.!same_type_as(this,rec)
   end subroutine draw2

   subroutine write_sq(this,results,i)
   class (square):: this
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

   subroutine write_sq2(results,i,o,this)
   class (square) :: this
   integer i
   integer, optional :: o
   logical results(:)
   type(rectangle) :: rec
   type(square) :: sq
   results(i) = .true.!extends_type_of(this,rec)
   end subroutine write_sq2

end module

program p
USE CHECK_MOD
use shape_mod


logical l 
logical results(6)
logical expect(6)
class(square),allocatable :: s
class(shape),allocatable :: sh
type(rectangle) :: r

results = .false.
expect = .true.

allocate(s)
s%the_length = 1000
s%color = 1
call s%write2(results=results,i=1,o=0)
call s%write(results,2)
call s%draw(results,3)

call s%rectangle%write2(results=results,i=4,o=1);
call draw2(s%rectangle,results,5)

allocate(sh)
call sh%write(results,6)

call check(results,expect,6)

end


