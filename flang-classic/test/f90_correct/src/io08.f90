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
 interface READ(FORMATTED)
   module procedure pwf
 end interface

 contains

 subroutine pwf(dtv, unit, iotype, vlist, iostat, iomsg)
 class(person), intent(inout) :: dtv
 integer, intent(in) :: unit
 character(len=*),intent(in) :: iotype
 integer, intent(in) :: vlist(:)
 integer, intent(out) :: iostat
 character (len=*), intent(inout) :: iomsg
 
 character(len=9) :: pfmt
 
! write(pfmt, '(a,i2,a,i2,a)' ) &
!   '(a', vlist(1), ',i', vlist(2), ')'

! write (unit,  *) dtv%name, dtv%age


 read (unit, *) dtv%name, dtv%age
 end subroutine

end module

 use person_module
 logical rslt(5), expect(5)
 integer id, members
 type(person) :: chairman

 chairman%name='myname'
 chairman%age=40
 id = 99
 rslt = .false.
 expect = .true.

 open(11, file='io08.inp')

 read(11, *, err=99)  id, chairman, members
 rslt(1) = .true.

 99  continue
 
 rslt(2) = id .eq. 2
 rslt(3) = chairman%name .eq. 'abc'
 rslt(4) = chairman%age .eq. 27 
 rslt(5) = members .eq. 1

 call check(rslt, expect, 5)

 end


