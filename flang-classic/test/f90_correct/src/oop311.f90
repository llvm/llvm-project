! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod_gen
implicit none
private
type, public :: v
integer, allocatable :: r(:)
class(v), pointer :: cpy
contains
procedure :: my_status
end type
type, public, extends(v) :: w
!integer,allocatable :: tag
integer tag
end type

type, public :: t
integer info
class(v),allocatable :: comp
end type

contains
logical function my_status(this)
class(v) :: this
!print *, 'status = ',allocated(this%r)
my_status = allocated(this%r)
end function

end module

program p
USE CHECK_MOD
use mod_gen
class(t), allocatable :: obj
class(t), allocatable :: obj2
real rr(10)
real rr2(10)
logical rslt(6)
logical expect(6)


rslt = .false.
expect = .true.

allocate(obj)
allocate(w::obj%comp)

allocate(obj%comp%r(10))
do i=1,10
obj%comp%r(i) = i
rr(i) = i+10
rr2(i) = i+20
enddo

select type(o=>obj%comp)
type is (w)
o%tag = 999
end select

allocate(obj%comp%cpy,source=obj%comp)
allocate(obj2,source=obj)

do i=1,10
obj2%comp%cpy%r(i) = obj2%comp%cpy%r(i) + 10
enddo

do i=1,10
obj%comp%r(i) = obj%comp%r(i) + 20
enddo

!deallocate(obj%comp%r)
!deallocate(obj%comp)

rslt(1) = obj2%comp%my_status()
rslt(2) = obj2%comp%cpy%my_status()

select type(o=>obj2%comp%cpy)
type is (w)
o%tag = 1234
rslt(3) = o%tag .eq. 1234
rslt(4) = all(o%r .eq. rr)
end select

select type(o=>obj%comp)
type is (w)
rslt(5) = o%tag .eq. 999
rslt(6) = all(o%r .eq. rr2)
end select

call check(rslt,expect,6)

end
