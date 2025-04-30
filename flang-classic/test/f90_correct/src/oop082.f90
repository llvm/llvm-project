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
	procedure,pass(this) :: write2 => write_sq2

end type square
contains

  subroutine write_shape(this,results,i)
   class (shape) :: this
   integer results(:)
   integer i
   type(shape) :: sh
   results(i) = same_type_as(sh,this)
   end subroutine write_shape

   subroutine write_rec(this,results,i)
   class (rectangle) :: this
   integer results(:)
   integer i
   type(shape) :: sh
   results(i) = same_type_as(sh,this)
   end subroutine write_rec

   integer function draw_shape(this,results,i)
   class (shape) :: this
   integer results(:)
   integer i
   type(shape)::sh
   type(square)::sq
   results(i) = extends_type_of(sq,this)
   draw_shape = same_type_as(this,sh)
	print *, 'DRAW_SH'
   end function draw_shape

   integer function draw_rectangle(this,results,i)
   class (rectangle) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   draw_rectangle = same_type_as(this,rec)
	print *, 'DRAW_REC'
   end function draw_rectangle

   subroutine write_sq(this,results,i)
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq

   integer function draw_sq(this,results,i)
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   type(square)::sq
   results(i) = extends_type_of(this,rec)
   draw_sq = same_type_as(this,sq)
   print *, 'DRAW_SQ'
   end function draw_sq

   subroutine write_sq2(i,this,results)
   class (square) :: this
   integer i 
   integer results(:)
   type(rectangle) :: rec

   results(i) = extends_type_of(this,rec)
   end subroutine write_sq2


end module shape_mod

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
call s%write2(1,results)
call s%write(results,2)
results(5) = s%draw(results,3)
allocate(rectangle::s2)
results(6) = s2%draw(results,7)


allocate(sh)
call sh%write(results,4)
results(8) = sh%draw(results,9)

call check(results,expect,9)

end


