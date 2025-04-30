! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Tests F2003 defined I/O (recursive read)

module person_module
 logical rslt(10), expect(10)
 integer :: cnt
 type :: person
  character(len=20) :: name
  integer :: age
 contains
  procedure :: rf
  procedure :: wf
  generic :: READ(FORMATTED) => rf
  generic :: WRITE(FORMATTED) => wf
 end type

 contains

 recursive subroutine rf(dtv, unit, iotype, vlist, iostat, iomsg)
 class(person), intent(inout) :: dtv
 integer, intent(in) :: unit
 character(len=*),intent(in) :: iotype
 integer, intent(in) :: vlist(:)
 integer, intent(out) :: iostat
 character (len=*), intent(inout) :: iomsg
 
 character(len=9) :: pfmt
 
 read (unit, *, iostat=iostat, err=99) dtv%name, dtv%age
 if (iostat .eq. 0) then
   cnt = cnt + 1
   rslt(cnt) = dtv%age .eq. 40+(cnt-1) 
   read(unit, *) dtv
 endif
99 continue 
 end subroutine

subroutine wf(dtv, unit, iotype, vlist, iostat, iomsg)
 class(person), intent(inout) :: dtv
 integer, intent(in) :: unit
 character(len=*),intent(in) :: iotype
 integer, intent(in) :: vlist(:)
 integer, intent(out) :: iostat
 character (len=*), intent(inout) :: iomsg

 character(len=9) :: pfmt

 write (unit, *) dtv%name, dtv%age
 end subroutine


end module

 use person_module
 integer id, members
 type(person) :: chairman

 chairman%name='myname'
 chairman%age=40
 id = 99
 rslt = .false.
 expect = .true.

 open(11, file='io10.output', status='replace')
 do i=1,10
   write(11, *) chairman
   chairman%age = chairman%age + 1
 enddo

 cnt = 0
 open(11, file='io10.output', position='rewind')
 read(11, *)  chairman
 
 close(11)

 call check(rslt, expect, 10)

 end


