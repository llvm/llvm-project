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
integer tag
end type

type, public :: t
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
real rr1(10)
real rr2(10)
logical expect(9)
logical rslt(9)

rslt = .false.
expect = .true.

allocate(obj)
!print *, allocated(obj%comp)
rslt(1) = .not. allocated(obj%comp)
allocate(w::obj%comp)
!print *, allocated(obj%comp)
rslt(2) = allocated(obj%comp)

allocate(obj%comp%r(10))
do i=1,10
obj%comp%r(i) = i
!print *, obj%comp%r(i)
rr1(i) = i+10
rr2(i) = i+20
enddo

select type(o=>obj%comp)
type is (w)
o%tag = 999
!print *, o
rslt(3) = o%tag .eq. 999
type is (v)
!print *, o
rslt(3) = .false.
end select

allocate(obj%comp%cpy,source=obj%comp)
allocate(obj2,source=obj)

do i=1,10
obj2%comp%cpy%r(i) = obj2%comp%cpy%r(i) + 10
enddo

do i=1,10
obj%comp%r(i) = obj%comp%r(i) + 20
enddo

rslt(4) = all(obj2%comp%cpy%r .eq. rr1)
rslt(5) = all(obj%comp%r .eq. rr2)

deallocate(obj%comp%r)
deallocate(obj%comp)

rslt(6) = obj2%comp%my_status()
rslt(7) = obj2%comp%cpy%my_status()

select type(o=>obj2%comp%cpy)
type is (w)
rslt(8) = o%tag .eq. 999
rslt(9) = all(o%r .eq. rr1)
end select

call check(rslt,expect,9)


end
