!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   Transfer intrinsic with derived type MOLD
module fred
  integer, parameter :: i1_kind = selected_int_kind(2)
  type, public :: byte_type
    private
    integer(i1_kind) :: data
  end type
contains
  subroutine put_char (data, data_pos, string)
    type(byte_type) :: data(*)
    integer :: data_pos
    character*(*) :: string
    data(data_pos+1:data_pos+len(string)) = transfer(string,data(1:0))
    data_pos = data_pos + len(string)
  end subroutine put_char
  subroutine init(data)
   type(byte_type)::data(:)
   data(:)%data = 0
  end subroutine
  subroutine copy(data,array,n)
   type(byte_type)::data(:)
   integer::array(:)
   do i = 1,n
    array(i) = data(i)%data
   enddo
  end subroutine
end module fred
program p
 use fred
 type(byte_type):: d1(20)
 integer p1
 integer result(21),expect(21)
 data expect/97,32,108,111,110,103,101,114,32,115,116,114,105,110,103,&
	0,0,0,0,0,15/
 call init(d1)
 p1 = 0
 call put_char(d1,p1,'a longer string')
 call copy(d1,result,20)
 result(21) = p1
 call check(result,expect,21)
end
