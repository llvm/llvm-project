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
	procedure :: write => write_shape 
	procedure :: draw => draw_shape
	procedure,nopass :: area => area_shape
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
contains
        procedure :: write => write_rec
	procedure :: draw => draw_rectangle
end type rectangle

type, extends (rectangle) :: square
contains
        procedure :: draw => draw_sq
	procedure :: write => write_sq
	procedure,pass(this) :: write2 => write_sq2
	procedure :: write3 => write_sq3

end type square
contains

  subroutine area_shape(x,y,results,expect,i)
  integer x
  integer y
  integer expect
  integer results(:)
  integer i
  results(i) = ((x*y) .eq. expect)
  end subroutine area_shape

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

   subroutine draw_shape(this,results,i)
   class (shape) :: this
   integer results(:)
   integer i
   print *, 'draw shape!'
   end subroutine draw_shape

   subroutine draw_rectangle(this,results,i)
   class (rectangle) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine draw_rectangle

   subroutine write_sq(this,results,i)
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq

   subroutine draw_sq(this,results,i)
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine draw_sq

   subroutine write_sq2(i,this,results)
   class (square) :: this
   integer i 
   integer results(:)
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq2

   integer function write_sq3(this)
   class (square) :: this
   type(rectangle) :: rec
   write_sq3 = extends_type_of(this,rec)
   end function 



end module shape_mod

program p
USE CHECK_MOD
use shape_mod

logical l 
integer results(6)
integer expect(6)
data expect /.true.,.true.,.true.,.true.,.true.,.true./
data results /.false.,.false.,.false.,.false.,.false.,.false./
class(square),allocatable :: s
class(shape),allocatable :: sh
type(rectangle) :: r
type(square),allocatable :: sq
integer a

allocate(s)
call s%write2(1,results)
call s%write(results,2)
call s%draw(results,3)

allocate(sq)
results(5) = sq%write3

allocate(sh)
call write_shape(sh,results,4)

call sh%area(6,5,results,30,6)

call check(results,expect,6)


end


