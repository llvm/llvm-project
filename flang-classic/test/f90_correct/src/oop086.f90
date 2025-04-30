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
end type rectangle

type, extends (rectangle) :: square
contains
        procedure :: draw => draw_sq
	procedure,pass(this) :: write => write_sq
	procedure,pass(this) :: write_sq2

end type square
   interface
   integer function draw_shape(this,results,i)
   import shape 
   class (shape) :: this
   integer results(:)
   integer i
   end function draw_shape
   end interface
   interface
   integer function draw_rectangle(this,results,i) RESULT(dr)
   import rectangle 
   class (rectangle):: this
   integer results(:)
   integer i
   end function draw_rectangle
   end interface
   interface
   subroutine write_sq(this,results,i)
   import square 
   class (square) :: this
   integer results(:)
   integer i
   end subroutine write_sq
   end interface
   interface
   integer function draw_sq(this,results,i) RESULT(ds)
   import square 
   class (square) :: this
   integer results(:)
   integer i
   end function draw_sq
   end interface
   interface
   subroutine write_sq2(i,results,this)
   import  square
   class (square) :: this
   integer i
   integer results(:)
   end subroutine write_sq2
   end interface
   interface
   subroutine write_shape(this,results,i)
   import shape 
   class (shape) :: this
   integer results(:)
   integer i
   end subroutine write_shape
   end interface
   interface
   subroutine write_rec(this,results,i)
   import rectangle 
   class (rectangle) :: this
   integer results(:)
   integer i
   end subroutine write_rec
   end interface

end module shape_mod

  subroutine write_shape(this,results,i) 
   use :: shape_mod, except => write_shape
   class (shape) :: this
   integer results(:)
   integer i
   type(shape) :: sh
   results(i) = same_type_as(sh,this)
	print *, 'WRITE_SHAPE'
   end subroutine write_shape

   subroutine write_rec(this,results,i)
   use :: shape_mod, except => write_rec
   class (rectangle) :: this
   integer results(:)
   integer i
   type(shape) :: sh
   results(i) = same_type_as(sh,this)
   end subroutine write_rec

   integer function draw_shape(this,results,i) RESULT(draw_shape)
   use :: shape_mod, except => draw_shape
   class (shape) :: this
   integer results(:)
   integer i
   integer r
   type(shape)::sh
   type(square)::sq
   results(i) = extends_type_of(sq,this)
   draw_shape = same_type_as(this,sh)
   end function draw_shape

   integer function draw_rectangle(this,results,i) RESULT(dr)
   use :: shape_mod, except => draw_rectangle
   class (rectangle) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   dr = same_type_as(this,rec)
   end function draw_rectangle

   subroutine write_sq(this,results,i)
   use :: shape_mod, except => write_sq
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
	print *, 'WRITE_SQ'
   end subroutine write_sq

   integer function draw_sq(this,results,i) RESULT(ds)
   use :: shape_mod, except => draw_sq
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   type(square)::sq
   results(i) = extends_type_of(this,rec)
   ds = same_type_as(this,sq)
   end function draw_sq

   subroutine write_sq2(i,results,this)
   use :: shape_mod, except => write_sq2
   class (square) :: this
   integer i 
   integer results(:)
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq2


program p
USE CHECK_MOD
use shape_mod

logical l 
integer results(9)
integer expect(9)
class(square),allocatable :: s
class(shape),allocatable :: sh
class(rectangle),allocatable::rec
class(shape),allocatable :: s2
type(rectangle) :: r

integer i

results = .false.
expect = .true.

allocate(s)
s%the_length = 1000
call s%write_sq2(1,results)
call s%write(results,2)
results(5) = s%draw(results,3)
allocate(rectangle::s2)
results(6) = s2%draw(results,7)

allocate(s2)
call s2%write(results,4)
results(8) = s2%draw(results,9)

call check(results,expect,9)

end


