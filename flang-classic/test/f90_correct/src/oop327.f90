! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       
! Tests final subroutines

module shape_mod

integer :: sq_count = 0 
integer :: rec_count = 0
integer :: sq_r1_count = 0
integer :: sq_r2_count = 0

type shape
        integer :: color
        logical :: filled
        integer :: x
        integer :: y
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
	contains
	final :: final_rect
end type rectangle

type, extends (rectangle) :: square
	!class(shape), allocatable :: sh
	contains
	procedure :: foo => bar
	final :: final_square, final_rankone, final_rank2
end type square

type composite
       type(square),allocatable :: sq
       type(rectangle) :: r
end type

contains

subroutine final_rect(this)
type(rectangle) :: this
!print *, 'final_rec called!',this%color
rec_count = rec_count + 1
end subroutine

subroutine bar(this)
class(square) :: this
!print *, 'bar called'
end subroutine

subroutine final_square(sq) 
type(square) sq
!print *, 'final procedure sq%color is ',sq%color
sq_count = sq_count + sq%color
end subroutine

subroutine final_rankone(sq)
type(square) sq(*)
!print *, 'final_rankone called'
sq_r1_count = sq_r1_count + 1
end subroutine

subroutine final_rank2(sq)
type(square) sq(1,*)
!print *, 'final_rank2 called'
sq_r2_count = sq_r2_count + 1
end subroutine

end module shape_mod

subroutine foo
use shape_mod

class(square), allocatable :: sh

allocate(sh)
sh%color = 99
deallocate(sh)
!print *, 'leaving foo'

end subroutine foo

subroutine nothing
use shape_mod
class(shape),allocatable :: ss
integer i,j


class(square), allocatable :: sh1(:)
class(square), allocatable :: sh2(:,:)

allocate(square::ss)
ss%color = 256
allocate(sh1(10))
allocate(sh2(2,2))

do i = 1, 10
  sh1(i)%color = 0
enddo

do i=1,2
 do j=1,2
  sh2(i,j)%color = 0
 enddo
enddo

deallocate(ss)
deallocate(sh1)
deallocate(sh2)

end subroutine

subroutine something
use shape_mod
type(composite), allocatable :: c
!type(composite)::c

allocate(c)
allocate(c%sq)
c%sq%color = 101

!print *, 'deallocate c%sq'
deallocate(c%sq)
!print *, 'deallocate c'
deallocate(c)
end subroutine

program prg
USE CHECK_MOD
use shape_mod
interface
subroutine something
end subroutine
subroutine nothing
end subroutine
end interface

logical rslt(4)
logical  expect(4)

rslt = .false.
expect = .true.

!print *, '********calling foo'
call foo
!print *, '********foo called'

!print *, '********calling nothing'
call nothing
!print *, '********nothing called'

!print *, '********calling something'
call something
!print *, '********something called'

!print *, sq_count, rec_count, sq_r1_count, sq_r2_count

rslt = .false.
expect = .true.
rslt(1) = sq_count .eq. 456
rslt(2) = rec_count .eq. 18
rslt(3) = sq_r1_count .eq. 1
rslt(4) = sq_r2_count .eq. 1

call check(rslt,expect,4)

end


