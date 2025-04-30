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
	class(shape),allocatable :: sq
end type rectangle

type, extends (rectangle) :: square
real, allocatable :: r(:)
end type square

end module shape_mod

program p
USE CHECK_MOD
use shape_mod
logical l 
logical results(8)
logical expect(8)
type(rectangle),allocatable :: r
type(rectangle),allocatable :: r2
real, allocatable :: rr(:)

expect = .true.
results = .false.

results(1) = .not. allocated(r)
allocate(r)
results(2) = allocated(r)
results(3) = .not. allocated(r%sq)

allocate(square::r%sq)
select type(o=>r%sq)
class is (square)
allocate(o%r(10))

do i = 1, size(o%r)
o%r(i) = i
enddo

allocate(rr(size(o%r)),source=o%r)

results(4) = all(rr .eq. o%r)

allocate(r2,source=r)
select type(o2=>r2%sq)
class is (square)
results(5) = all(o2%r .eq. rr)

results(6) = all(o2%r .eq. o%r)

do i = 1, size(o2%r)
o2%r(i) = i + 50
rr(i) = i + 50
enddo

results(7) = .not. all(o2%r .eq. o%r)

!deallocate(r2)

results(8) = .not. all(o%r .eq. rr)
end select
end select

call check(results,expect,8)

end


