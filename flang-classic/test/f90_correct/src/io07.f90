! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests F2003 defined I/O

! based on figure 17.2 from "Modern Fortran explained" book
module person_module
 type :: person
  character(len=20) :: name
  integer :: age
 end type
 interface WRITE(FORMATTED)
   module procedure pwf
 end interface

 interface READ(FORMATTED)
   module procedure rwf
 end interface


 contains

 subroutine pwf(dtv, unit, iotype, vlist, iostat, iomsg)
 class(person), intent(in) :: dtv
 integer, intent(in) :: unit
 character(len=*),intent(in) :: iotype
 integer, intent(in) :: vlist(:)
 integer, intent(out) :: iostat
 character (len=*), intent(inout) :: iomsg
 
 character(len=8) :: pfmt
 
 write(pfmt, '(a,i2,a,i1,a)' ) &
   '(a', vlist(1), ',i', vlist(2), ')'
 write (unit, fmt=pfmt, iostat=iostat) dtv%name, dtv%age
 end subroutine

subroutine rwf(dtv, unit, iotype, vlist, iostat, iomsg)
 class(person), intent(inout) :: dtv
 integer, intent(in) :: unit
 character(len=*),intent(in) :: iotype
 integer, intent(in) :: vlist(:)
 integer, intent(out) :: iostat
 character (len=*), intent(inout) :: iomsg

 character(len=8) :: pfmt

 write(pfmt, '(a,i2,a,i1,a)' ) &
   '(a', vlist(1), ',i', vlist(2), ')'
 read (unit, fmt=pfmt) dtv%name, dtv%age   ! this fails
 !read (unit, fmt='(a10,i3)') dtv%name, dtv%age  ! this works
 end subroutine

end module

 use person_module
 integer id, members, id2, members2
 type(person) :: chairman
 type(person) :: candidate
 logical rslt(4), expect(4)

 id = 99
 members = 12345
 chairman%age = 50
 chairman%name = 'John_Smith'

 open(10, file='io07.output', form='formatted', status='replace')  
 write(10, fmt="(i2, X, dt(10,3), X, i5)" ) id, chairman, members
 close(10)

 id2 = 0
 members2 = 0
 open(11, file='io07.output', form='formatted', status='old')
 read(11, fmt="(i2, X, dt(10,3), X, i5)" ) id2, candidate, members2

 close(11)

 !print *, id, candidate%name, candidate%age, members

  expect = .true.

  rslt(1) = id .eq. id2
  rslt(2) = members .eq. members2
  rslt(3) = candidate%name .eq. chairman%name
  rslt(4) = candidate%age .eq. chairman%age

  call check(rslt,expect,4)

 end
  
