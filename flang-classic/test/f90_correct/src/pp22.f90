! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests the F95 NULL() intrinsic
!

program pp22

  parameter(N=23)

  integer :: result(N) = -1
  integer :: expect(N) 
  data expect / 12*0,1,2*0,1, 0,1,0,1,0,1,0/

  integer, target :: i1
  integer, target, dimension(5) :: iArr1
  
  integer, pointer :: iPtr1 => NULL()
  integer, pointer, dimension(:) :: iArrPtr1 => NULL()

  type t1
    integer, pointer :: t1_iptr1
    integer, pointer, dimension(:) :: t1_iarrptr1
    integer :: t1_i1
  end type
  type (t1) :: t1_inst = t1(NULL(), NULL(), 2)
  type (t1) :: t1_inst2
  data t1_inst2 / t1(NULL(), NULL(),1) /
 
  type t2
    type (t1) :: t2_t1inst
  end type
 
  type (t2) :: t2_inst = t2(t1(NULL(), NULL(),1))
  type (t2) :: t2_inst2
  data t2_inst2 / t2(t1(NULL(), NULL(),1)) /

  type t3
    integer, pointer :: t3_iptr1=>NULL()
    integer, pointer, dimension(:) :: t3_iarrptr1=>NULL()
    integer :: t3_i1
  end type

  type (t3) :: t3_inst1
  type (t3), pointer :: t3_ptrInst1

  if (associated(t1_inst%t1_iarrptr1) ) then
    result(1) = 1
  else
    result(1) = 0
  endif

  if (associated(t1_inst%t1_iptr1) ) then
    result(2) = 1
  else
    result(2) = 0
  endif

  if (associated(t2_inst%t2_t1inst%t1_iarrptr1) ) then
    result(3) = 1
  else
    result(3) = 0
  endif

  if (associated(t2_inst%t2_t1inst%t1_iptr1) ) then
    result(4) = 1
  else
    result(4) = 0
  endif

  if (associated(t1_inst2%t1_iarrptr1) ) then
    result(5) = 1
  else
    result(5) = 0
  endif

  if (associated(t1_inst2%t1_iptr1) ) then
    result(6) = 1
  else
    result(6) = 0
  endif

  if (associated(t2_inst2%t2_t1inst%t1_iarrptr1) ) then
    result(7) = 1
  else
    result(7) = 0
  endif

  if (associated(t2_inst2%t2_t1inst%t1_iptr1) ) then
    result(8) = 1
  else
    result(8) = 0
  endif

  if (associated(t3_inst1%t3_iarrptr1) ) then
    result(9) = 1
  else
    result(9) = 0
  endif

  if (associated(t3_inst1%t3_iptr1) ) then
    result(10) = 1
  else
    result(10) = 0
  endif

  allocate(t3_ptrInst1)
  if (associated(t3_ptrinst1%t3_iarrptr1) ) then
    result(11) = 1
  else
    result(11) = 0
  endif

  if (associated(t3_ptrinst1%t3_iptr1) ) then
    result(12) = 1
  else
    result(12) = 0
  endif

  if (associated(t3_ptrinst1) ) then
    result(13) = 1
  else
    result(13) = 0
  endif

  if (associated(iArrPtr1) ) then
    result(14) = 1
  else
    result(14) = 0
  endif

  if (associated(iPtr1) ) then
    result(15) = 1
  else
    result(15) = 0
  endif

  iArrPtr1=>iArr1
  if (associated(iArrPtr1) ) then
    result(16) = 1
  else
    result(16) = 0
  endif

  iArrPtr1=>NULL()
  if (associated(iArrPtr1) ) then
    result(17) = 1
  else
    result(17) = 0
  endif

  iPtr1=>i1
  if (associated(iPtr1) ) then
    result(18) = 1
  else
    result(18) = 0
  endif

  iPtr1=>NULL()
  if (associated(iPtr1) ) then
    result(19) = 1
  else
    result(19) = 0
  endif

  t1_inst%t1_iarrptr1=>iArr1
  if (associated(t1_inst%t1_iarrptr1) ) then
    result(20) = 1
  else
    result(20) = 0
  endif

  t1_inst%t1_iarrptr1=>NULL()
  if (associated(t1_inst%t1_iarrptr1) ) then
    result(21) = 1
  else
    result(21) = 0
  endif

  t1_inst%t1_iptr1=>i1
  if (associated(t1_inst%t1_iptr1) ) then
    result(22) = 1
  else
    result(22) = 0
  endif

  t1_inst%t1_iptr1=>NULL()
  if (associated(t1_inst%t1_iptr1) ) then
    result(23) = 1
  else
    result(23) = 0
  endif

  call check(result, expect, N)
end program
