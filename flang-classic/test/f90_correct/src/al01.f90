! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Extended ALLOCATABLE attribute tests
!   assignments involving allocatable arrays
!
!  TEST MUST BE COMPILED WITH 2003 ALLOCATABLE ASSIGNMENT SEMANTICS
!  ENABLED  (currently x,54,1)

module tstmod
 implicit none
   integer,allocatable :: m_alloc_i

   type t0
    integer :: i2(4)
    integer :: lb_i2, ub_i2
   end type

   type t
     integer :: i
     integer, allocatable :: alloc_iary(:)
   end type

  type l
    type(t), allocatable :: alloc_t_inst
    type(t), pointer :: t_ptr
  end type

   type t1
    integer, allocatable :: i2(:)
    integer :: lb_i2, ub_i2
   end type

  type t2
    type(t1), allocatable :: t1_f(:)
    integer :: lb_t1_f, ub_t1_f
  end type

  type t3
    integer :: lb_t3_f, ub_t3_f
    type(t1), allocatable :: t1_f
  end type

contains

subroutine print_t0(header, t0_i)
  character(len=*)::header
  type(t0):: t0_i

  print *,header
  print *, t0_i%lb_i2, t0_i%ub_i2
  print *,header, "%i2(1,4)@",loc(t0_i%i2), ":"
  print *, t0_i%i2
end subroutine

subroutine print_t0_array(header, t0_arr)
  character(len=*)::header
  type(t0):: t0_arr(:)
  integer ::i

  do i = lbound(t0_arr, 1), ubound(t0_arr, 1)
    call print_t0(header//"("//char(i+ichar('0'))//")",t0_arr(i))
  end do
end subroutine

subroutine init_t2(t2_i, lb, ub, init) 
  type(t2) :: t2_i
  integer :: lb
  integer :: ub
  integer :: init
  integer :: extnt 
  integer :: i

  extnt = ub - lb + 1
  allocate( t2_i%t1_f(lb:ub))
  t2_i%lb_t1_f = lb
  t2_i%ub_t1_f = ub

  do i = 1,extnt
    call init_t1(t2_i%t1_f(i), 1, i+1, init+i)
  end do
end subroutine

subroutine init_t1(t1_i, lb, ub, init)
  type(t1) :: t1_i
  integer :: lb
  integer :: ub
  integer :: init
  integer :: extnt 
  integer :: i

  extnt = ub - lb + 1
  allocate( t1_i%i2(lb:ub))
  t1_i%lb_i2 = lb
  t1_i%ub_i2 = ub

  do i = 1,extnt
    t1_i%i2(i) = (ub-lb+1) * i +init
  end do
end subroutine

subroutine init_t(t_i, ival, init)
    type(t) :: t_i
    integer :: ival
    integer :: init
    integer :: lb
    integer :: ub
    integer :: extnt
    integer :: i

  lb = lbound(t_i%alloc_iary,1)
  ub = ubound(t_i%alloc_iary,1)
  if( allocated(t_i%alloc_iary) ) then 
    do i = 1,extnt
      t_i%alloc_iary(i) = (ub-lb+1) * i +init
    end do
     t_i%i = ival
  end  if

end subroutine

subroutine print_t(header, t_i)
  character(len=*)::header
  type(t):: t_i
  print *,header
  
  print *,header,"%i=", t_i%i
  if( allocated(t_i%alloc_iary) ) then
    print *,header, "%alloc_iary(",char(lbound(t_i%alloc_iary,1)+ichar('0')),":", &
                           char(ubound(t_i%alloc_iary,1)+ichar('0')),")@",loc(t_i%alloc_iary), ":"
    print *, t_i%alloc_iary
  else
    print *,  header, "%alloc_iary not allocated"
  endif
  
end subroutine

subroutine print_t1(header, t1_i)
  character(len=*)::header
  type(t1):: t1_i

  print *,header
  print *, t1_i%lb_i2, t1_i%ub_i2
  if( allocated(t1_i%i2) ) then
    print *,header, "%i2(",char(lbound(t1_i%i2,1)+ichar('0')),":", &
                           char(ubound(t1_i%i2,1)+ichar('0')),")@",loc(t1_i%i2), ":"
    print *, t1_i%i2
  else
    print *,  header, "%i2 not allocated"
  endif

end subroutine

subroutine print_t1_array(header, t1_arr)
  character(len=*)::header
  type(t1):: t1_arr(:)
  integer ::i

  do i = lbound(t1_arr, 1), ubound(t1_arr, 1)
    call print_t1(header//"("//char(i+ichar('0'))//")",t1_arr(i))
  end do
end subroutine

subroutine print_t2(header, t2_i)
  character(len=*)::header
  type(t2):: t2_i

  print *,header
  print *, t2_i%lb_t1_f, t2_i%ub_t1_f
  if( allocated(t2_i%t1_f) ) then
    call print_t1_array(header//'%t1_f', t2_i%t1_f)
  else
    print *,  header, "%t1_f not allocated"
  endif
end subroutine

subroutine print_t2_array(header, t2_arr)
  character(len=*)::header
  type(t2):: t2_arr(:)
  integer ::i

  do i = lbound(t2_arr, 1), ubound(t2_arr, 1)
    print *,header, "%t1_f(",char(lbound(t2_arr(i)%t1_f,1)+ichar('0')),":", &
                             char(ubound(t2_arr(i)%t1_f,1)+ichar('0')),")@",loc(t2_arr(i)%t1_f), ":"

    print *,header//"("//char(i+ichar('0'))//")"
    print *, t2_arr(i)%lb_t1_f, t2_arr(i)%ub_t1_f
    if( allocated(t2_arr(i)%t1_f) ) then
      call print_t1_array(header//"("//char(i+ichar('0'))//")%t1_f", t2_arr(i)%t1_f)
    else
      print *,  header//"("//char(i+ichar('0'))//")%t1_f not allocated"
    endif
  end do
end subroutine

subroutine print_t3(header, t3_i)
  character(len=*)::header
  type(t3):: t3_i

  print *,header
  print *, t3_i%lb_t3_f, t3_i%ub_t3_f
  if( allocated(t3_i%t1_f) ) then
   call print_t1(header//"t3_i%t1_f", t3_i%t1_f)
  else
    print *,  header, "%t1_f not allocated"
  endif

end subroutine

subroutine print_l( header, l_i)
  character(len=*)::header
  type(l):: l_i

  print *,header
  if( allocated(l_i%alloc_t_inst) ) then
    call print_t(header//"%alloc_t_inst", l_i%alloc_t_inst)
  else
    print *, header//"%alloc_t_inst not allocated"
  end if

  if( associated(l_i%t_ptr ) ) then
    call print_t(header//"%t_ptr", l_i%t_ptr)
  else
    print *, header//"%t_ptr not associated"
  end if
end subroutine

end module
subroutine init_2dim_array(arr, d1_strt, d2_strt)
 integer :: arr(:,:) 
 integer :: d1_strt, d2_strt
 integer :: i, j

  do i = lbound(arr,1), ubound(arr,1)
    do j = lbound(arr,2), ubound(arr,2)
      arr(i,j) = i*10 +j
    end do
  end do
 
 
end subroutine

program p
 use tstmod
 implicit none
 
 interface
  subroutine init_2dim_array(arr, d1_strt, d2_strt)
   integer :: arr(:,:)
   integer :: d1_strt, d2_strt
  end subroutine
 end interface

  integer, parameter :: N = 103
  integer :: curtest
  integer :: result(N)
  integer :: expect(N) =  &
   (/1,   3,   1,   4,   1,   5, &
     1,   5,   1,   3,   1,   4, &
    -1,   0,   1,   3,   1,   4, &
    -1,   1,   4,   4,   3,   2, &
     1,  -1,   1,   3,   1,   4, &
    -1,   1,   4,  -1,   1,   4, &
    -1,   1,   5,  -1,   1,   3, &
    -1,   1,   3,  -1,   1,   2, &
     1,   3,   1,   4,  -1,  -1, &
    -1,   1,   2,   1,   3,   1, &
     4,  -1,  -1,  -1,  -1,  -1, &
    -1,   0,  -1,  -1,  -1, &
    -1,  -1,  -1,  -1,  -1,   0, &
    -1,  -1,  -1,  -1,  -1, &
    -1,  -1,  -1,   0,  -1, &
    -1,  -1,  -1,  -1,  -1,  -1, &
    -1,   0,  -1,  -1,  -1,  -1, &
    -1,  -1,  -1,  -1 /)


  integer, allocatable :: i_allocArr1(:,:)
  integer, allocatable :: i_allocArr2(:,:)
  integer :: i_Arr2(3,4)
  type(t0), allocatable :: t0_alloc1
  type(t0), allocatable :: t0_alloc2
  type(t0) :: t0_2
  type(t1), allocatable :: t1_alloc1
  type(t1), allocatable :: t1_alloc2
  type(t1) :: t1_2
   type(t1), allocatable :: t1_allocArr1(:)
   type(t1), allocatable :: t1_allocArr2(:)
   type(t1) :: t1_Arr2(3)
  type(t2), allocatable :: t2_alloc1
  type(t2), allocatable :: t2_alloc2
  type(t2), allocatable :: t2_allocArr1(:)
  type(t2), allocatable :: t2_allocArr2(:)
   integer::i, j

  curtest = 1
 call init_2dim_array(i_Arr2, 1,1)
!  print *,"i_Arr2:"
!  print *, i_Arr2

 allocate(i_allocArr2(3,4))
 call init_2dim_array(i_allocArr2, 1,1)
  result(curtest) = lbound(i_allocArr2, 1)
  result(curtest+1) = ubound(i_allocArr2, 1)
  result(curtest+2) = lbound(i_allocArr2, 2)
  result(curtest+3) = ubound(i_allocArr2, 2)
  curtest = curtest + 4
!  print *,"addr(i_allocArr2)=",loc(i_allocArr2)
!  print *,"i_allocArr2:"
!  print *, i_allocArr2

  allocate(i_allocArr1(5,5))
  result(curtest) = lbound(i_allocArr1, 1)
  result(curtest+1) = ubound(i_allocArr1, 1)
  result(curtest+2) = lbound(i_allocArr1, 2)
  result(curtest+3) = ubound(i_allocArr1, 2)
  curtest= curtest + 4
!  print *,"addr(i_allocArr2)=",loc(i_allocArr2)
!  print *,"i_allocArr2:"
!  print *, i_allocArr2

 i_allocArr1 = i_allocArr2
  result(curtest) = lbound(i_allocArr1, 1)
  result(curtest+1) = ubound(i_allocArr1, 1)
  result(curtest+2) = lbound(i_allocArr1, 2)
  result(curtest+3) = ubound(i_allocArr1, 2)
  result(curtest+4) = all(i_allocArr1 .eq. i_allocArr2)
  result(curtest+5) = loc(i_allocArr1) .eq. loc(i_allocArr2)
  curtest= curtest + 6

!  print *,"i_allocArr2:"
!  print *, i_allocArr2
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1)
!  print *, "i_allocArr1:"
!  print *, i_allocArr1
  call init_2dim_array(i_allocArr1, -1,-1)
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1), "AFTER RE-INIT"
!  print *, "i_allocArr1:"
!  print *, i_allocArr1

  i_allocArr1 = i_Arr2
  result(curtest) = lbound(i_allocArr1, 1)
  result(curtest+1) = ubound(i_allocArr1, 1)
  result(curtest+2) = lbound(i_allocArr1, 2)
  result(curtest+3) = ubound(i_allocArr1, 2)
  result(curtest+4) = all(i_allocArr1 .eq. i_Arr2)
  curtest= curtest + 5

!  print *,"addr(i_allocArr1)=",loc(i_allocArr1)
!  print *, "i_allocArr1:"
!  print *, i_allocArr1
call init_2dim_array(i_allocArr1, 2,3)
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1), "AFTER RE-INIT"
!  print *, "i_allocArr1:"
!  print *, i_allocArr1

  t0_2 = t0((/4,3,2,1/), 1, 4)
!  print *, t0_2%i2, t0_2%lb_i2, t0_2%ub_i2
  t0_alloc2 = t0_2
  result(curtest) = t0_alloc2%lb_i2
  result(curtest+1) = t0_alloc2%ub_i2
  curtest= curtest + 2

  result(curtest:curtest+3) =  t0_2%i2
  curtest= curtest + 4

!  print *, "t0_2:"
!  print *, "t0_alloc2@", loc(t0_alloc2),":"
!  print *, t0_alloc2%i2, t0_alloc2%lb_i2, t0_alloc2%ub_i2
  t0_alloc1 = t0_alloc2
!  print *, "t0_alloc1@", loc(t0_alloc1),":"
!  print *, t0_alloc1%i2, t0_alloc1%lb_i2, t0_alloc1%ub_i2

!  print *,"addr(i_allocArr1)=",loc(i_allocArr1)
!  print *, "i_allocArr1:"
!  print *, i_allocArr1

  i_allocArr1 = i_allocArr2
  result(curtest) = allocated(i_allocArr1)
  curtest= curtest + 1

!  print *,"i_allocArr2:"
!  print *, i_allocArr2
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1)
!  print *, "i_allocArr1:"
!  print *, i_allocArr1
  call init_2dim_array(i_allocArr1, -1,-1)
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1), "AFTER RE-INIT"
!  print *, "i_allocArr1:"
!  print *, i_allocArr1

  i_allocArr1 = i_Arr2
  result(curtest) = lbound(i_allocArr1, 1)
  result(curtest+1) = ubound(i_allocArr1, 1)
  result(curtest+2) = lbound(i_allocArr1, 2)
  result(curtest+3) = ubound(i_allocArr1, 2)
  result(curtest+4) = all(i_allocArr1 .eq. i_Arr2)
  curtest= curtest + 5

!  print *,"addr(i_allocArr1)=",loc(i_allocArr1)
!  print *, "i_allocArr1:"
!  print *, i_allocArr1
  call init_2dim_array(i_allocArr1, 2,3)
!  print *,"addr(i_allocArr1)=",loc(i_allocArr1), "AFTER RE-INIT"
!  print *, "i_allocArr1:"
!  print *, i_allocArr1

  t0_2 = t0((/4,3,2,1/), 1, 4)
!  print *, "t0_2:"
!  print *, t0_2%i2, t0_2%lb_i2, t0_2%ub_i2
  t0_alloc2 = t0_2
  result(curtest) = lbound(t0_alloc2%i2,1)
  result(curtest+1) = ubound(t0_alloc2%i2,1)
  result(curtest+2) = all(t0_alloc2%i2 .eq. t0_2%i2)
  curtest= curtest + 3
!  print *, "t0_alloc2@", loc(t0_alloc2),":"
!  print *, t0_alloc2%i2, t0_alloc2%lb_i2, t0_alloc2%ub_i2
  t0_alloc1 = t0_alloc2
  result(curtest) = lbound(t0_alloc1%i2,1)
  result(curtest+1) = ubound(t0_alloc1%i2,1)
  result(curtest+2) = all(t0_alloc2%i2 .eq. t0_alloc1%i2)
  curtest= curtest + 3
!  print *, "t0_alloc1@", loc(t0_alloc1),":"
!  print *, t0_alloc1%i2, t0_alloc1%lb_i2, t0_alloc1%ub_i2

  call init_t1(t1_2, 1,5, 10)
  t1_alloc1 = t1_2
  result(curtest) = lbound(t1_alloc1%i2,1)
  result(curtest+1) = ubound(t1_alloc1%i2,1)
  result(curtest+2) = all( t1_alloc1%i2 .eq. t1_2%i2)
  curtest= curtest + 3
!  call print_t1("t1_2", t1_2)
!  call print_t1("t1_alloc1", t1_alloc1)

  allocate(t1_alloc2)
  call init_t1(t1_alloc2, 1,3, 21)
  t1_alloc1 = t1_alloc2
  result(curtest) = lbound(t1_alloc1%i2,1)
  result(curtest+1) = ubound(t1_alloc1%i2,1)
  result(curtest+2) = all( t1_alloc1%i2 .eq. t1_alloc2%i2)
  curtest= curtest + 3
!  call print_t1("t1_alloc1", t1_alloc1)
  t1_2 = t1_alloc1
  result(curtest) = lbound(t1_2%i2,1)
  result(curtest+1) = ubound(t1_2%i2,1)
  result(curtest+2) = all( t1_alloc1%i2 .eq. t1_2%i2)
  curtest= curtest + 3
!  call print_t1("t1_2", t1_2)

  allocate(t1_allocArr2(1:3))
  call init_t1(t1_allocArr2(1), 1,2, 0)
  call init_t1(t1_allocArr2(2), 1,3, 4)
  call init_t1(t1_allocArr2(3), 1,4, 8)
!  call print_t1_array("t1_allocArr2", t1_allocArr2)

  t1_allocArr1 = t1_allocArr2
  result(curtest) = lbound(t1_allocArr1(1)%i2,1)
  result(curtest+1) = ubound(t1_allocArr1(1)%i2,1)
  result(curtest+2) = lbound(t1_allocArr1(2)%i2,1)
  result(curtest+3) = ubound(t1_allocArr1(2)%i2,1)
  result(curtest+4) = lbound(t1_allocArr1(3)%i2,1)
  result(curtest+5) = ubound(t1_allocArr1(3)%i2,1)
  result(curtest+6) = all(t1_allocArr1(1)%i2 .eq. t1_allocArr2(1)%i2)
  result(curtest+7) = all(t1_allocArr1(2)%i2 .eq. t1_allocArr2(2)%i2)
  result(curtest+8) = all(t1_allocArr1(1)%i2 .eq. t1_allocArr2(1)%i2)
  curtest= curtest + 9

!  call print_t1_array("t1_allocArr1", t1_allocArr1)

  call init_t1(t1_Arr2(1), 1,2, 0)
  call init_t1(t1_Arr2(2), 1,3, 4)
  call init_t1(t1_Arr2(3), 1,4, 8)
!  call print_t1_array("t1_Arr2", t1_Arr2)

  t1_allocArr1 = t1_Arr2
  result(curtest) = lbound(t1_allocArr1(1)%i2,1)
  result(curtest+1) = ubound(t1_allocArr1(1)%i2,1)
  result(curtest+2) = lbound(t1_allocArr1(2)%i2,1)
  result(curtest+3) = ubound(t1_allocArr1(2)%i2,1)
  result(curtest+4) = lbound(t1_allocArr1(3)%i2,1)
  result(curtest+5) = ubound(t1_allocArr1(3)%i2,1)
  result(curtest+6) = all(t1_allocArr1(1)%i2 .eq. t1_Arr2(1)%i2)
  result(curtest+7) = all(t1_allocArr1(2)%i2 .eq. t1_Arr2(2)%i2)
  result(curtest+8) = all(t1_allocArr1(3)%i2 .eq. t1_Arr2(3)%i2)
  curtest= curtest + 9

!  call print_t1_array("t1_allocArr1", t1_allocArr1)

  t1_Arr2 = t1_allocArr1 
  result(curtest) = all(t1_allocArr1(1)%i2 .eq. t1_Arr2(1)%i2)
  result(curtest+1) = all(t1_allocArr1(2)%i2 .eq. t1_Arr2(2)%i2)
  result(curtest+2) = all(t1_allocArr1(3)%i2 .eq. t1_Arr2(3)%i2)
  curtest= curtest + 3


!  call print_t1_array("t1_Arr2", t1_Arr2)

  allocate( t2_alloc2 )
  call init_t2(t2_alloc2, 1,2, 3)
!  call print_t2("t2_alloc2", t2_alloc2)
  t2_alloc1 = t2_alloc2
  result(curtest) = loc(t2_alloc1) .eq. loc(t2_alloc2)
  result(curtest+1) = lbound(t2_alloc1%t1_f,1) .eq. lbound(t2_alloc2%t1_f,1)
  result(curtest+2) = ubound(t2_alloc1%t1_f,1) .eq. ubound(t2_alloc2%t1_f,1)
  result(curtest+3) = lbound(t2_alloc1%t1_f(1)%i2,1) .eq. lbound(t2_alloc2%t1_f(1)%i2,1)
  result(curtest+4) = ubound(t2_alloc1%t1_f(1)%i2,1) .eq. ubound(t2_alloc2%t1_f(1)%i2,1)
  result(curtest+5) = lbound(t2_alloc1%t1_f(2)%i2,1) .eq. lbound(t2_alloc2%t1_f(2)%i2,1)
  result(curtest+6) = ubound(t2_alloc1%t1_f(2)%i2,1) .eq. ubound(t2_alloc2%t1_f(2)%i2,1)
  result(curtest+7) = all(t2_alloc1%t1_f(1)%i2 .eq. t2_alloc2%t1_f(1)%i2)
  result(curtest+8) = all(t2_alloc1%t1_f(2)%i2 .eq. t2_alloc2%t1_f(2)%i2)
  curtest= curtest + 9
  
!  call print_t2("t2_alloc1", t2_alloc1)

   allocate( t2_allocArr2(1:3) )
   call init_t2(t2_allocArr2(1), 1,2, 2)
   call init_t2(t2_allocArr2(2), 1,2, 3)
   call init_t2(t2_allocArr2(3), 1,2, 4)
!   call print_t2_array("t2_allocArr2", t2_allocArr2)
   t2_allocArr1 = t2_allocArr2
  result(curtest) = loc(t2_allocArr1(1)) .eq. loc(t2_allocArr2(1))
  result(curtest+1) = lbound(t2_allocArr1(1)%t1_f,1) .eq. lbound(t2_allocArr2(1)%t1_f,1)
  result(curtest+2) = ubound(t2_allocArr1(1)%t1_f,1) .eq. ubound(t2_allocArr2(1)%t1_f,1)
  result(curtest+3) = lbound(t2_allocArr1(1)%t1_f(1)%i2,1) .eq. lbound(t2_allocArr2(1)%t1_f(1)%i2,1)
  result(curtest+4) = ubound(t2_allocArr1(1)%t1_f(1)%i2,1) .eq. ubound(t2_allocArr2(1)%t1_f(1)%i2,1)
  result(curtest+5) = lbound(t2_allocArr1(1)%t1_f(2)%i2,1) .eq. lbound(t2_allocArr2(1)%t1_f(2)%i2,1)
  result(curtest+6) = ubound(t2_allocArr1(1)%t1_f(2)%i2,1) .eq. ubound(t2_allocArr2(1)%t1_f(2)%i2,1)
  result(curtest+7) = all(t2_allocArr1(1)%t1_f(1)%i2 .eq. t2_allocArr2(1)%t1_f(1)%i2)
  result(curtest+8) = all(t2_allocArr1(1)%t1_f(2)%i2 .eq. t2_allocArr2(1)%t1_f(2)%i2)
  curtest= curtest + 9

  result(curtest) = loc(t2_allocArr1(2)) .eq. loc(t2_allocArr2(2))
  result(curtest+1) = lbound(t2_allocArr1(2)%t1_f,1) .eq. lbound(t2_allocArr2(2)%t1_f,1)
  result(curtest+2) = ubound(t2_allocArr1(2)%t1_f,1) .eq. ubound(t2_allocArr2(2)%t1_f,1)
  result(curtest+3) = lbound(t2_allocArr1(2)%t1_f(1)%i2,1) .eq. lbound(t2_allocArr2(2)%t1_f(1)%i2,1)
  result(curtest+4) = ubound(t2_allocArr1(2)%t1_f(1)%i2,1) .eq. ubound(t2_allocArr2(2)%t1_f(1)%i2,1)
  result(curtest+5) = lbound(t2_allocArr1(2)%t1_f(2)%i2,1) .eq. lbound(t2_allocArr2(2)%t1_f(2)%i2,1)
  result(curtest+6) = ubound(t2_allocArr1(2)%t1_f(2)%i2,1) .eq. ubound(t2_allocArr2(2)%t1_f(2)%i2,1)
  result(curtest+7) = all(t2_allocArr1(2)%t1_f(1)%i2 .eq. t2_allocArr2(2)%t1_f(1)%i2)
  result(curtest+8) = all(t2_allocArr1(2)%t1_f(2)%i2 .eq. t2_allocArr2(2)%t1_f(2)%i2)
  curtest= curtest + 9

  result(curtest) = loc(t2_allocArr1(3)) .eq. loc(t2_allocArr2(3))
  result(curtest+1) = lbound(t2_allocArr1(3)%t1_f,1) .eq. lbound(t2_allocArr2(3)%t1_f,1)
  result(curtest+2) = ubound(t2_allocArr1(3)%t1_f,1) .eq. ubound(t2_allocArr2(3)%t1_f,1)
  result(curtest+3) = lbound(t2_allocArr1(3)%t1_f(1)%i2,1) .eq. lbound(t2_allocArr2(3)%t1_f(1)%i2,1)
  result(curtest+4) = ubound(t2_allocArr1(3)%t1_f(1)%i2,1) .eq. ubound(t2_allocArr2(3)%t1_f(1)%i2,1)
  result(curtest+5) = lbound(t2_allocArr1(3)%t1_f(2)%i2,1) .eq. lbound(t2_allocArr2(3)%t1_f(2)%i2,1)
  result(curtest+6) = ubound(t2_allocArr1(3)%t1_f(2)%i2,1) .eq. ubound(t2_allocArr2(3)%t1_f(2)%i2,1)
  result(curtest+7) = all(t2_allocArr1(3)%t1_f(1)%i2 .eq. t2_allocArr2(3)%t1_f(1)%i2)
  result(curtest+8) = all(t2_allocArr1(3)%t1_f(2)%i2 .eq. t2_allocArr2(3)%t1_f(2)%i2)
  curtest= curtest + 9

!   call print_t2_array("t2_allocArrArr1(1)", t2_allocArrArr1(1))

  call check(result,expect, N)
end program
