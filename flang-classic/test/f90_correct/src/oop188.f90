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
	procedure :: func => shape_func
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
contains
        procedure,pass(this) :: write => write_rec
	procedure :: draw => draw_rectangle
	procedure :: func => rec_func
end type rectangle

type, extends (rectangle) :: square
contains
        procedure :: draw => draw_sq
	procedure :: write => write_sq
	procedure,pass(this) :: write2 => write_sq2
	procedure :: write3 => write_sq3
	procedure :: func => sq_func

end type square
contains

  subroutine write_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   type(shape) :: sh
   !results(i) = same_type_as(sh,this)
   select type(this)
   type is (shape)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine write_shape

   subroutine write_rec(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(shape) :: sh
   !results(i) = same_type_as(sh,this)
   select type(this)
   type is (rectangle)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine write_rec

   subroutine draw_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   print *, 'draw shape!'
   results(i) = .true.
   end subroutine draw_shape

   subroutine draw_rectangle(this,results,i)
   class (rectangle) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   select type(this)
   class is (rectangle)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine draw_rectangle

   subroutine write_sq(this,results,i)
   class (square) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   select type(this)
   class is (square)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine write_sq

   subroutine draw_sq(this,results,i)
   class (square) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   select type(this)
   class is (square)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine draw_sq

   subroutine write_sq2(i,this,results)
   class (square) :: this
   integer i 
   logical results(:)
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   select type(this)
   class is (square)
   results(i) = .true.
   class default
   results(i) = .false.
   end select
   end subroutine write_sq2

   logical function write_sq3(this)
   class (square) :: this
   type(rectangle) :: rec
   !write_sq3 = extends_type_of(this,rec)
   select type(this)
   class is (square)
   write_sq3 = .true.
   class default
   write_sq3 = .false.
   end select
   end function 

   logical function sq_func(this)
   class (square) :: this
   sq_func = .false.
   end function

   logical function rec_func(this)
   class (rectangle) :: this
   type(square) :: sq
   select type(this)
   type is (rectangle)
   rec_func = .true.
   class default
   rec_func = .false.
   end select
   
   end function

   logical function shape_func(this)
   class (shape) :: this
   shape_func = .false.
   end function

end module shape_mod

program p
USE CHECK_MOD
use shape_mod

logical l 
logical results(6)
logical expect(6)
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
results(5) = sq%write3()

allocate(sh)
call write_shape(sh,results,4)

results(6) = s%rectangle%func()

call check(results,expect,6)


end


