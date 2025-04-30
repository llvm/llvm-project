
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Extended ALLOCATABLE attribute tests
!   derived type constructors and array constructors assigned to ALLOCATABLEs
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
print*,lb,ub
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

program p
 use tstmod

 integer, parameter :: N = 169
 
 integer :: curtst
 integer :: result(N)
 integer :: expect(N)  = (/ &
 ! TEST 1-6
    0, 0, 0, 0, 0, 0, &
 ! TEST 7-12
    1, 4, 1, 2, 3, 4, &
 ! TEST 13-30
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, &
 ! TEST 31-48
    2, 5, 2, 3, 4, 5, 2, 5, 2, 3, 4, 5, 2, 5, 2, 3, 4, 5, &
 ! TEST 49-60
    3, 6, 3, 4, 5, 6, 4, 7, 4, 5, 6, 7, &
 ! TEST 61-68
    0, 0, 0, 1, 3, 1, 2, 3, &
 ! TEST 69-78
    0, 2, 0, 1, 2, 0, 2, 0, 1, 2, &
 ! TEST 79-86
    2, 3, 2, 3, 3, 5, 4, 5, &
 ! TEST 87-96
    1, 3, 1, 2, 3, 1, 3, 1, 2, 3, &
 ! TEST 97-99
    1, 3, -1, &
 ! TEST 100-114
    1, 3, 1, 2, 3, 1, 3, 1, 2, 3, 1, 3, 1, 2, 3, &
 ! TEST 115-123
    2, 3, 2, 3, 3, 5, 3, 4, 5, &
 ! TEST 124-128
    1, 0, 1, 2, -1, &
 ! TEST 129-134
    1, 0, 1, 2, 6, 6, &
 ! TEST 135-136
    8, 0, &
 ! TEST 137-138
    0, 0, &
 ! TEST 139-144
    6, 2, 1, 1, 2, 0, &
 ! TEST 145-150
    6, 2, 1, 1, 2, 0, &
 ! TEST 151-159
    -1, -2, -1, -3, 3, 1, 33, 22, 11, &
 ! TEST 160-166
    8, -1, 1, 2, 2, 1, 0, &
 ! TEST 167-169
    8, 0, -1  /)

 integer, target, allocatable :: l_iary(:)
 type(t0), allocatable :: t0_i1
 type(t0), allocatable :: t0_arr1(:)
 type(t), target :: t_inst
 type(t),pointer :: t_ptr
 type(t), allocatable :: t_allocarr(:)
 type(t1) :: t1_i1
 type(t1), allocatable :: t1_i2
 type(t1) :: t1_arr1(2) 
 type(t1) :: t1_arr2(2)
 type(t1), allocatable :: t1_arr3(:)
 type(l) :: list1
 type(t3) :: t3_alloci


  curtst = 1

! TEST 1-6
  allocate(t0_i1)
  t0_i1 = t0((/0,0,0,0/), 0,0)
  result(curtst) = t0_i1%lb_i2
  result(curtst+1) = t0_i1%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_i1%i2
  curtst = curtst + 4

! TEST 7-12
  deallocate(t0_i1)
  allocate(t0_i1)
  t0_i1 = t0((/1,2,3,4/), 1,4)
  result(curtst) = t0_i1%lb_i2
  result(curtst+1) = t0_i1%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_i1%i2
  curtst = curtst + 4

!  print *, "TEST ", curtst
! TEST 13-30
  allocate(t0_arr1(3))
  t0_arr1 = t0((/0,0,0,0/), 0,0)
  result(curtst) = t0_arr1(1)%lb_i2
  result(curtst+1) = t0_arr1(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(1)%i2
  curtst = curtst + 4

  result(curtst) = t0_arr1(2)%lb_i2
  result(curtst+1) = t0_arr1(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(2)%i2
  curtst = curtst + 4


  result(curtst) = t0_arr1(3)%lb_i2
  result(curtst+1) = t0_arr1(3)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(3)%i2
  curtst = curtst + 4


!  print *, "TEST ", curtst
! TEST 31-48
  t0_arr1 = t0((/2,3,4,5/), 2,5)
  result(curtst) = t0_arr1(1)%lb_i2
  result(curtst+1) = t0_arr1(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(1)%i2
  curtst = curtst + 4

  result(curtst) = t0_arr1(2)%lb_i2
  result(curtst+1) = t0_arr1(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(2)%i2
  curtst = curtst + 4


  result(curtst) = t0_arr1(3)%lb_i2
  result(curtst+1) = t0_arr1(3)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(3)%i2
  curtst = curtst + 4

!  call print_t0_array("t0_arr1", t0_arr1)

!  print *, "TEST ", curtst
! TEST 49-60
  deallocate(t0_arr1)
  allocate(t0_arr1(2))
  t0_arr1 = (/ t0((/3,4,5,6/), 3,6), t0((/4,5,6,7/), 4,7) /) 
!  t0_arr1 = t0((/2,3,4,5/), 2,5)
  result(curtst) = t0_arr1(1)%lb_i2
  result(curtst+1) = t0_arr1(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(1)%i2
  curtst = curtst + 4

  result(curtst) = t0_arr1(2)%lb_i2
  result(curtst+1) = t0_arr1(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+3) = t0_arr1(2)%i2
  curtst = curtst + 4

!  call print_t0_array("t0_arr1", t0_arr1)

!  print *, "TEST ", curtst
! TEST 61-68
  t1_arr2(2) = t1((/1,2,3/), 1, 3)
  result(curtst) =  t1_arr2(1)%lb_i2
  result(curtst+1) =  t1_arr2(1)%ub_i2
  result(curtst+2) = allocated(t1_arr2(1)%i2)
  curtst = curtst + 3

  result(curtst) =  t1_arr2(2)%lb_i2
  result(curtst+1) =  t1_arr2(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr2(2)%i2
  curtst = curtst + 3


!  call print_t1_array("t1_arr2", t1_arr2)

!  print *, "TEST ", curtst
! TEST 69-78
  t1_arr2 = t1((/0,1,2/), 0, 2)

  result(curtst) =  t1_arr2(1)%lb_i2
  result(curtst+1) =  t1_arr2(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr2(1)%i2
  curtst = curtst + 3

  result(curtst) =  t1_arr2(2)%lb_i2
  result(curtst+1) =  t1_arr2(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr2(2)%i2
  curtst = curtst + 3

!  call print_t1_array("t1_arr2", t1_arr2)

!  print *, "TEST ", curtst
! TEST 79-86
  t1_arr2 = (/ t1((/2,3/), 2, 3), t1((/4,5/), 3, 5) /) 

  result(curtst) =  t1_arr2(1)%lb_i2
  result(curtst+1) =  t1_arr2(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+1) = t1_arr2(1)%i2
  curtst = curtst + 2

  result(curtst) =  t1_arr2(2)%lb_i2
  result(curtst+1) =  t1_arr2(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+1) = t1_arr2(2)%i2
  curtst = curtst + 2

!  call print_t1_array("t1_arr2", t1_arr2)

!  print *, "TEST ", curtst
! TEST 87-96
  t1_arr2 = t1((/1,2,3/), 1, 3)

  result(curtst) =  t1_arr2(1)%lb_i2
  result(curtst+1) =  t1_arr2(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr2(1)%i2
  curtst = curtst + 3

  result(curtst) =  t1_arr2(2)%lb_i2
  result(curtst+1) =  t1_arr2(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr2(2)%i2
  curtst = curtst + 3

!  call print_t1_array("t1_arr2", t1_arr2)


!  print *, "TEST ", curtst
! TEST 97-99
!!  t1_arr3 = t1((/1,2,3/), 1, 3)
  result(curtst) =  t1_arr2(1)%lb_i2
  result(curtst+1) =  t1_arr2(1)%ub_i2
  result(curtst+2) = allocated( t1_arr2(1)%i2)
  curtst = curtst + 3

!  print *,"t1_arr3(",lbound(t1_arr3,1),":", ubound(t1_arr3,1),")"
!  call print_t1_array("t1_arr3", t1_arr3)


!  print *, "TEST ", curtst
! TEST 100-114
  allocate(t1_arr3(3))
  t1_arr3 = t1((/1,2,3/), 1, 3)

  result(curtst) =  t1_arr3(1)%lb_i2
  result(curtst+1) =  t1_arr3(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr3(1)%i2
  curtst = curtst + 3

  result(curtst) =  t1_arr3(2)%lb_i2
  result(curtst+1) =  t1_arr3(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr3(2)%i2
  curtst = curtst + 3

  result(curtst) =  t1_arr3(3)%lb_i2
  result(curtst+1) =  t1_arr3(3)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr3(3)%i2
  curtst = curtst + 3

!  call print_t1_array("t1_arr3", t1_arr3)


!  print *, "TEST ", curtst
! TEST 115-123
  allocate(t1_arr3(2))
  t1_arr3 = (/ t1((/2,3/), 2, 3), t1((/3,4,5/), 3, 5) /)

  result(curtst) =  t1_arr3(1)%lb_i2
  result(curtst+1) =  t1_arr3(1)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+1) = t1_arr3(1)%i2
  curtst = curtst + 2

  result(curtst) =  t1_arr3(2)%lb_i2
  result(curtst+1) =  t1_arr3(2)%ub_i2
  curtst = curtst + 2
  result(curtst:curtst+2) = t1_arr3(2)%i2
  curtst = curtst + 3

!  call print_t1_array("t1_arr3", t1_arr3)

!  print *, "TEST ", curtst
! TEST 124-128
!!  t_allocarr = t(6,(/1,2/))  !! no alloc/relloc checks for array = scalar
  result(curtst)   =  1 !! lbound(t_allocarr,1)
  result(curtst+1) =  0 !! ubound(t_allocarr,1)
  curtst = curtst + 2
!  print *,"t_allocarr(",lbound(t_allocarr,1),":", ubound(t_allocarr,1),")"

  allocate(t_allocarr(2))
  t_allocarr = t(6,(/1,2/))
  result(curtst) =  lbound(t_allocarr,1)
  result(curtst+1) =  ubound(t_allocarr,1)
  result(curtst+2)  = allocated(t_allocarr(1)%alloc_iary)
  curtst = curtst + 3

!  print *,"t_allocarr(",lbound(t_allocarr,1),":", ubound(t_allocarr,1),")"
!  print *,t_allocarr


!  print *, "TEST ", curtst
! TEST 129-134
!!  l_iary = 5  !! no alloc/relloc checks for array = scalar
  result(curtst)   = 1  !! lbound(l_iary,1)
  result(curtst+1) = 0  !! ubound(l_iary,1)
  curtst = curtst + 2

!  print *,"l_iary(",lbound(l_iary,1),":", ubound(l_iary,1),")"
!  print *,l_iary

  allocate(l_iary(2))
  l_iary = 6
  result(curtst) =  lbound(l_iary,1)
  result(curtst+1) =  ubound(l_iary,1)
  curtst = curtst + 2

  result(curtst:curtst+1) =  l_iary
  curtst = curtst + 2

!  print *,"l_iary(",lbound(l_iary,1),":", ubound(l_iary,1),")"
!  print *,l_iary


!  print *, "TEST ", curtst
! TEST 135-136
  t_inst = t(8,null())
  result(curtst) = t_inst%i
  result(curtst+1) = allocated(t_inst%alloc_iary)
  curtst = curtst + 2

!  call print_t("t_inst",t_inst)


!  print *, "TEST ", curtst
! TEST 137-138
  list1 = l(null(),null() )
  result(curtst) = allocated(list1%alloc_t_inst)
  result(curtst+1) = associated(list1%t_ptr)
  curtst = curtst + 2

!  call print_l("list1", list1)


!  print *, "TEST ", curtst
! TEST 139-144
  list1 = l(t_allocarr(1),null() )
  result(curtst) = list1%alloc_t_inst%i
  result(curtst+1) = ubound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+2) = lbound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+3:curtst+4) = list1%alloc_t_inst%alloc_iary
  result(curtst+5) = associated(list1%t_ptr)
  curtst = curtst + 6

!  call print_l("list1", list1)

  
!  print *, "TEST ", curtst
! TEST 145-150
  list1 = l(t_allocarr(1),t_ptr )
  result(curtst) = list1%alloc_t_inst%i
  result(curtst+1) = ubound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+2) = lbound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+3:curtst+4) = list1%alloc_t_inst%alloc_iary
  result(curtst+5) = associated(list1%t_ptr)
  curtst = curtst + 6

!  call print_l("list1", list1)


!  print *, "TEST ", curtst
! TEST 151-159
  t1_i1 =  t1((/33,22,11/), -1, -3) 

  t3_alloci = t3(-1,-2, t1_i1 )
  result(curtst) = t3_alloci%lb_t3_f
  result(curtst+1) = t3_alloci%ub_t3_f

  result(curtst+2) = t3_alloci%t1_f%lb_i2
  result(curtst+3) = t3_alloci%t1_f%ub_i2

  result(curtst+4) = ubound(t3_alloci%t1_f%i2,1)
  result(curtst+5) = lbound(t3_alloci%t1_f%i2,1)
  curtst = curtst + 6

  result(curtst:curtst+2) = t3_alloci%t1_f%i2
  curtst = curtst + 3

!  call print_t3("t3_alloci", t3_alloci)

  
!  print *, "TEST ", curtst
! TEST 160-166
  list1 = l( t(8,(/2,1/)),t_ptr )

  result(curtst) = list1%alloc_t_inst%i
  result(curtst+1) = allocated(list1%alloc_t_inst%alloc_iary)
  result(curtst+2) = lbound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+3) = ubound(list1%alloc_t_inst%alloc_iary,1)
  result(curtst+4:curtst+5) = list1%alloc_t_inst%alloc_iary
  result(curtst+6) = associated(list1%t_ptr)
  curtst = curtst + 7

!  print *, "TEST ", curtst
! TEST 167-169
  t_ptr => t_inst
  list1 = l( t(8,null()),t_ptr )
  result(curtst) = list1%alloc_t_inst%i
  result(curtst+1) = allocated(list1%alloc_t_inst%alloc_iary)
  result(curtst+2) = associated(list1%t_ptr)
  curtst = curtst + 3

!  call print_l("list1", list1)

  call check(result,expect, N)

end 

