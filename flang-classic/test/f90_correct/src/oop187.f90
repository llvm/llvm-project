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
	procedure :: write3 => write_sq3

end type square
contains

  subroutine write_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   type(shape) :: sh
   !results(i) = same_type_as(sh,this)
   select type(this)
   type is(shape)
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
   select type (this)
   type is (rectangle)
   results(i) = .false.
   class default
   results(i) = .true.
   end select
   end subroutine write_rec

   subroutine draw_shape(this,results,i)
   class (shape) :: this
   logical results(:)
   integer i
   results(i) = .false.
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
   class default
   !results(i) = extends_type_of(this,rec)
   results(i) = .true.
   end select
   end subroutine write_sq

   subroutine draw_sq(this,results,i)
   class (square) :: this
   logical results(:)
   integer i
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   results(i) = .true.
   end subroutine draw_sq

   subroutine write_sq2(i,this,results)
   class (square) :: this
   integer i 
   logical results(:)
   type(rectangle) :: rec
   !results(i) = extends_type_of(this,rec)
   results(i) = .true.
   end subroutine write_sq2

   logical function write_sq3(this)
   class (square) :: this
   type(rectangle) :: rec
   !write_sq3 = extends_type_of(this,rec)
   write_sq3 = .true.
   end function 

end module shape_mod

program p
USE CHECK_MOD
use shape_mod

logical l 
logical results(12)
logical expect(12)
class(square),allocatable :: s
class(shape),allocatable :: sh
type(rectangle) :: r
type(square),allocatable :: sq
class(square),allocatable :: sq2

results = .false.
expect = .true.

allocate(s)
call s%write2(1,results)
call s%write(results,2)
call s%draw(results,3)

allocate(sq)
results(5) = sq%write3()

allocate(sh)
call write_shape(sh,results,4)

deallocate(sh)
allocate(square::sh)
sh%filled = .false.

select type(sh)
class is(shape)
results(6) = .false.
class is(rectangle)
results(6) = .false.
class default
results(6) = .false.
class is (square)
results(6) = .true.
end select

select type(sh)
class is(shape)
results(7) = .false.
type is(shape)
results(7) = .false.
class is(rectangle)
results(7) = .false.
type is (rectangle)
results(7) = .false.
class default
results(7) = .false.
class is (square)
results(7) = .false.
type is (square)
sh%the_length=1000
results(7) = .true.
end select

select type(o=>sh)
type is (square)
results(9) = o%rectangle%the_length .eq. 1000
o%filled = .true.
results(8) = .true.
class is(shape)
results(8) = .false.
type is(shape)
results(8) = .false.
class is(rectangle)
results(8) = .false.
type is (rectangle)
results(8) = .false.
class default
results(8) = .false.
class is (square)
results(8) = .false.
end select

results(10) = sh%filled

allocate(sq2)
select type(o=>sq2)
type is(square)
results(11) = .true.
o%the_width = 999
end select

results(12) = sq2%rectangle%the_width .eq. 999

call check(results,expect,12)

end


