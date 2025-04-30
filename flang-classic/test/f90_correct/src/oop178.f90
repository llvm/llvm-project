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
end type rectangle

type, extends (rectangle) :: square
	integer :: eq
end type square

end module shape_mod

program p
USE CHECK_MOD
use shape_mod
logical l 
logical results(6)
logical expect(6)
class(square),allocatable :: s
class(rectangle),allocatable :: r
class(shape),allocatable :: sh

results = .false.
expect = .true.

allocate(r)
r%the_width = 987
r%color = -1
r%filled = 1
r%x = 100
r%y = 200
allocate(sh, source=r)

select type (sh)
type is (rectangle)
sh%the_length = 777
r%x = 200
class default
end select
sh%y = 300


select type (sh)
type is (rectangle)
results(1) = sh%the_length .eq. 777
results(2) = sh%the_width .eq. 987
results(3) = sh%color .eq. -1
results(4) = sh%x .eq. 100
results(5) = sh%y .eq. 300
results(6) = sh%filled .eq. 1
class is (rectangle)
end select

call check(results,expect,6)

end


