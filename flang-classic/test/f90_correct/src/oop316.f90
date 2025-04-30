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
	class(square),pointer :: sq => null()
end type rectangle

type, extends (rectangle) :: square
end type square

end module shape_mod

program p
USE CHECK_MOD
use shape_mod
logical l 
logical results(5)
logical expect(5)
class(rectangle),allocatable :: r

expect = .true.
results = .false.

results(1) = .not. allocated(r)
allocate(r)
results(2) = allocated(r)
results(3) = .not. associated(r%sq)
allocate(r%sq)
results(4) = associated(r%sq)
deallocate(r)
results(5) = .not. allocated(r)

call check(results,expect,5)

end


