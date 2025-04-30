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

interface
 subroutine write_sq(this,results,i)
   import square 
   class (square) :: this
   integer results(:)
   integer i
end subroutine
end interface
interface
  subroutine draw_sq(this,results,i)
   import square 
   class (square) :: this
   integer results(:)
   integer i
end subroutine
end interface
interface
subroutine write_sq2(results,i,this)
   import  square
   class (square) :: this
   integer i
   integer results(:)
end subroutine
end interface
interface
 subroutine write_shape(this,results,i)
   import shape 
   class (shape) :: this
   integer results(:)
   integer i
end subroutine
end interface
interface
subroutine draw_shape(this,results,i)
   import shape 
   class (shape) :: this
   integer results(:)
   integer i
end subroutine
end interface
interface
subroutine draw_rectangle(this,results,i)
   import rectangle 
   class (rectangle) :: this
   integer results(:)
   integer i
   end subroutine draw_rectangle
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
   end subroutine write_shape

   subroutine write_rec(this,results,i)
   use :: shape_mod, except => write_rec
   class (rectangle) :: this
   integer results(:)
   integer i
   type(shape) :: sh
   results(i) = same_type_as(sh,this)
   end subroutine write_rec

   subroutine draw_shape(this,results,i)
   use :: shape_mod, except => draw_shape
   class (shape) :: this
   integer results(:)
   integer i
   print *, 'draw shape!'
   end subroutine draw_shape

   subroutine draw_rectangle(this,results,i)
   use :: shape_mod, except => draw_rectangle
   class (rectangle):: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine draw_rectangle

   subroutine write_sq(this,results,i)
   use :: shape_mod, except => write_sq
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq

   subroutine draw_sq(this,results,i)
   use :: shape_mod, except => draw_sq
   class (square) :: this
   integer results(:)
   integer i
   type(rectangle) :: rec
   results(i) = extends_type_of(this,rec)
   end subroutine draw_sq

   subroutine write_sq2(results,i,this)
   use :: shape_mod, except => write_sq2
   class (rectangle) :: this
   integer i
   integer results(:)
   type(rectangle) :: rec
   type(square) :: sq
   results(i) = extends_type_of(this,rec)
   end subroutine write_sq2


program p
USE CHECK_MOD
use shape_mod

logical l 
integer results(4)
integer expect(4)
data expect /.true.,.true.,.true.,.true./
data results /.false.,.false.,.false.,.false./
class(square),allocatable,target :: s
class(shape),allocatable,target :: sh
class(square),pointer :: ptr
class(shape),pointer :: ps
type(rectangle) :: r

allocate(s)
ptr => s
s%the_length = 1000
s%color = 1
call ptr%write2(results,1)
call ptr%write(results,2)
call ptr%draw(results,3)

allocate(sh)
ps => sh
call ps%write(results,4)

call check(results,expect,4)

end


