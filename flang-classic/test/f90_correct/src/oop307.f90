! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod_gen
implicit none
private
type, public :: v
integer, allocatable :: r(:)
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
real rr(10)
logical rslt(5)
logical expect(5)

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
rr(i) = i
enddo

select type(o=>obj%comp)
type is (w)
o%tag = 999
rslt(3) = .true.
!print *, o
type is (v)
!print *, o
end select

allocate(obj2,source=obj)
deallocate(obj%comp%r)
deallocate(obj%comp)
rslt(4) = obj2%comp%my_status()
select type(o=>obj2%comp)
type is (w)
!print *, o
rslt(5) = all(rr .eq. o%r)
type is (v)
!print *, o
end select

call check(rslt,expect,5)


end
