! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod_gen
implicit none
private
type, public :: v
integer r(10)
end type
type, public, extends(v) :: w
integer tag
end type

type, public :: t
class(v),allocatable :: comp
end type
end module

program p
USE CHECK_MOD
use mod_gen
class(t), allocatable :: obj
class(t), allocatable :: obj2
logical rslt(7)
logical expect(7)
real rr(10)

rslt = .false.
expect = .true.

allocate(obj)
!print *, allocated(obj%comp)
rslt(1) = .not. allocated(obj%comp)
allocate(w::obj%comp)
!print *, allocated(obj%comp)
rslt(2) = allocated(obj%comp)



do i=1,10
obj%comp%r(i) = i
rr(i) = i
enddo

select type(o=>obj%comp)
type is (w)
o%tag = 999
rslt(3) = all(o%r .eq. rr)
!print *, o
end select

allocate(obj2,source=obj)
deallocate(obj%comp)
rslt(4) = .not. allocated(obj%comp)
rslt(5) = allocated(obj2%comp)


select type(o=>obj2%comp)
type is (w)
!print *, o
rslt(6) = o%tag .eq. 999
rslt(7) = all(o%r .eq. rr)
end select

call check(rslt,expect,7)

end
