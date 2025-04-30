! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Same as oop315.f90, but this tests sourced allocation with multiple
! allocatable targets (e.g., allocate(a,b, source=c)

module my_mod

type a
integer, pointer :: x(:)
contains
procedure :: my_dealloc => dealloc_a
end type

type, extends(a) :: b
real, allocatable :: y(:)
contains
procedure :: my_dealloc => dealloc_b
end type

type, extends(b) :: c
integer, allocatable :: tag
integer, pointer :: aux
contains
procedure :: my_dealloc => dealloc_c
end type

contains
subroutine dealloc_c(this)
class(c) :: this
deallocate(this%tag)
deallocate(this%aux)
call this%b%my_dealloc()
end subroutine

subroutine dealloc_b(this)
class(b) :: this
deallocate(this%y)
call this%a%my_dealloc()
end subroutine

subroutine dealloc_a(this)
class(a) :: this
deallocate(this%x)
end subroutine


end module


program p
!USE CHECK_MOD
use my_mod
logical rslt(12), expect(12)
integer i,j
class(a), allocatable :: obj, obj2

rslt = .false.
expect = .true.

allocate(c::obj)
allocate(c::obj2)

i = 12345
j = 999
select type(o2=>obj2)
type is (c)
select type (o=>obj)
type is (c)
allocate(o2%tag, o%tag,source=i)
allocate(o%aux,source=j)
allocate(o%y(10))
allocate(o%x(20))
rslt(1) = allocated(o%tag)
rslt(2) = associated(o%aux)
rslt(3) = allocated(o%y)
rslt(4) = associated(o%x)
rslt(10) = o%tag .eq. i
rslt(11) = o%aux .eq. j
rslt(12) = o2%tag .eq. o%tag
end select
end select

call obj%my_dealloc()

select type (o=>obj)
type is (c)
rslt(5) = .not. allocated(o%tag)
rslt(6) = .not. associated(o%aux)
rslt(7) = .not. allocated(o%y)
rslt(8) = .not. associated(o%x)
rslt(9) = allocated(obj)
end select

deallocate(obj)

call check(rslt,expect,12)

end




