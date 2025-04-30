! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Extended ALLOCATABLE attribute tests
!   allocatable arguments and return values
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

  if( allocated(t_i%alloc_iary) ) then 
    lb = lbound(t_i%alloc_iary,1)
    ub = ubound(t_i%alloc_iary,1)
    do i = lb,ub
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
function f1(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocScalar)
 use tstmod
 implicit none
  integer, intent(IN), allocatable :: argAllocScalar
  integer, allocatable :: argAllocIntArr(:)
  type(t), allocatable :: argAllocArr_t(:)
  integer, allocatable :: rsltAllocScalar
  integer :: tstRslt(:)
  integer :: tstNbr

  tstNbr = 1
  tstRslt(tstNbr) = allocated(rsltAllocScalar)

  allocate(rsltAllocScalar)

  tstNbr = tstNbr +1
  if( allocated(argAllocScalar) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif

  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end function

function f2(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocIntArr)
 use tstmod
 implicit none
  integer, intent(OUT), allocatable :: argAllocScalar
  integer, intent(IN), allocatable :: argAllocIntArr(:)
  type(t), intent(OUT),  allocatable :: argAllocArr_t(:)
  integer, allocatable :: rsltAllocIntArr(:)
  integer :: tstRslt(:)
  integer :: tstNbr

  tstNbr = 1
  tstRslt(tstNbr) = allocated(rsltAllocIntArr)
  
  allocate(rsltAllocIntArr(4))

  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end function


function f3(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocArr_t)
 use tstmod
 implicit none
  integer, intent(IN), allocatable :: argAllocScalar
  integer, intent(OUT), allocatable :: argAllocIntArr(:)
  type(t), intent(IN),  allocatable :: argAllocArr_t(:)
  type(t), allocatable :: rsltAllocArr_t(:)
  integer :: tstRslt(:)
  integer :: tstNbr

  tstNbr = 1
  tstRslt(tstNbr) = allocated(rsltAllocArr_t)

  allocate(rsltAllocArr_t(4))

  tstNbr = tstNbr +1
  if( allocated(argAllocScalar) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end function

subroutine s1(argAllocScalar, argAllocIntArr, argallocArr_t, tstRslt)
 use tstmod
 implicit none
  integer, allocatable :: argAllocScalar(:)
  integer, allocatable :: argAllocIntArr(:)
  type(t), allocatable :: argallocArr_t(:)
  integer :: tstRslt(:)
  integer :: tstNbr 

  tstNbr = 0

  tstNbr = tstNbr +1
  if( allocated(argAllocScalar) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif

  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end subroutine

subroutine s2(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt )
 use tstmod
 implicit none
  integer, allocatable, intent(OUT) :: argAllocScalar(:)
  integer, allocatable, intent(IN) :: argAllocIntArr(:)
  type(t), allocatable, intent(OUT) :: argallocArr_t(:)
  integer :: tstRslt(:)
  integer :: tstNbr
  logical :: ll

  tstNbr = 0

  tstNbr = tstNbr +1
  if( allocated(argAllocScalar) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif

  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end subroutine

subroutine s3(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt )
 use tstmod
 implicit none
  integer, allocatable, intent(IN) :: argAllocScalar(:)
  integer, allocatable, intent(OUT) :: argAllocIntArr(:)
  type(t), allocatable, intent(IN) :: argallocArr_t(:)
  integer :: tstRslt(:)
  integer :: tstNbr
  logical :: ll

  tstNbr = 0

  tstNbr = tstNbr +1
  if( allocated(argAllocScalar) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif

  tstNbr = tstNbr +1
  if( allocated(argAllocIntArr) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
    
  tstNbr = tstNbr +1
  if( allocated(argAllocArr_t) ) then
      tstRslt(tstNbr) = .TRUE.
  else
      tstRslt(tstNbr) = .FALSE.
  endif
end subroutine

program p
 use tstmod
 implicit none

 interface 
  function f1(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocScalar)
   use tstmod
    integer, intent(IN), allocatable :: argAllocScalar
    integer, allocatable :: argAllocIntArr(:)
    type(t), allocatable :: argAllocArr_t(:)
    integer, allocatable :: rsltAllocScalar
    integer :: tstRslt(:)
  end function
  
  function f2(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocIntArr)
   use tstmod
    integer, intent(IN), allocatable :: argAllocScalar
    integer, intent(IN), allocatable :: argAllocIntArr(:)
    type(t), intent(IN),  allocatable :: argAllocArr_t(:)
    integer, allocatable :: rsltAllocIntArr(:)
    integer :: tstRslt(:)
  end function
  
  function f3(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt) result(rsltAllocArr_t)
   use tstmod
    integer, intent(IN), allocatable :: argAllocScalar
    integer, intent(OUT), allocatable :: argAllocIntArr(:)
    type(t), intent(IN),  allocatable :: argAllocArr_t(:)
    type(t), allocatable :: rsltAllocArr_t(:)
    integer :: tstRslt(:)
  end function

  subroutine s1(argAllocScalar, argAllocIntArr, argallocArr_t, tstRslt )
   use tstmod
    integer, allocatable :: argAllocScalar
    integer, allocatable :: argAllocIntArr(:)
    type(t), allocatable :: argallocArr_t(:)
    integer :: tstRslt(:)
  end subroutine
  
  subroutine s2(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt )
   use tstmod
    integer, allocatable, intent(OUT) :: argAllocScalar
    integer, allocatable, intent(IN) :: argAllocIntArr(:)
    type(t), allocatable, intent(OUT) :: argallocArr_t(:)
    integer :: tstRslt(:)
  end subroutine

  subroutine s3(argAllocScalar, argAllocIntArr, argAllocArr_t, tstRslt )
   use tstmod
    integer, allocatable, intent(IN) :: argAllocScalar
    integer, allocatable, intent(OUT) :: argAllocIntArr(:)
    type(t), allocatable, intent(IN) :: argallocArr_t(:)
    integer :: tstRslt(:)
  end subroutine
 end interface

  integer, parameter :: N = 90
  integer :: result(N)
  integer :: expect(N)  = &
  (/ 0,  0, -1,  0,  0, -1, 0, -1, -1, &
!  TEST 10
     0, -1,  0, 0, -1, -1,  0,  0,  0, &
!  TEST 19
    -1,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0, 0, &
!  TEST 32
    -1,  0,  0, -1,  0, -1, -1,  0, &
!  TEST 40
     0, -1, -1, -1,  0, -1,  0,  0,  0, &
!  TEST 49
     0,  0,  0,  0,  0,  0, 0,  0,  0, &
!  TEST 58
    -1,  0,  0, 0, -1,  0,  0,  0, -1, 0,  0, &
!  TEST 69
     0, -1,  0,  0, 0, -1,  0,  0,  0, -1, 0, -1, &
!  TEST 81
    -1, -1,  0, -1, -1, -1,  0, -1,  0, -1 /)


  integer :: curtst
  logical :: a

  type(t), target::t_inst
  type(t), allocatable :: t_allocarr(:)
  type(t), allocatable :: t_allocarr2(:)
  type(l)::l_inst
  type(l), allocatable ::l_allocarr(:)
  integer, target, allocatable :: l_iary(:)
  integer, target, allocatable :: l_iary2(:)
  integer, allocatable :: ialloc
  integer, allocatable :: ialloc2
  integer :: addrRetVal

  curtst = 1

! print *,"TEST 1"

  result(curtst) = allocated(t_inst%alloc_iary)
  if( a ) then
!    print *, curtst, ". FAIL: not allocated(t_inst%alloc_iary)"
  else
!    print *, curtst, ". PASS: not allocated(t_inst%alloc_iary)"
  end if
  curtst = curtst +1

  result(curtst) = associated(l_inst%t_ptr)
  if( associated(l_inst%t_ptr) ) then
!    print *, curtst, ". FAIL: not allocated(l_inst%t_ptr)"
  else
!    print *,curtst, ". PASS: not allocated(l_inst%t_ptr)"
  end if
  curtst = curtst + 1

  allocate(t_inst%alloc_iary(5))
  result(curtst) = allocated(t_inst%alloc_iary)
  if( result(curtst) ) then
!    print *, curtst, ". PASS: allocated(t_inst%alloc_iary)"
  else
!    print *, curtst, ". FAIL: allocated(t_inst%alloc_iary)"
  end if
  curtst = curtst + 1

  result(curtst) = allocated(t_allocarr)
  if( result(curtst) ) then
!     print *,curtst, ". FAIL: allocated(t_allocarr)"
  else 
!     print *, curtst, ". PASS: allocated(t_allocarr)"
  end if
  curtst = curtst + 1

  t_allocarr2 = t_allocarr
  result(curtst) = allocated(t_allocarr2)
  if( result(curtst) ) then
!     print *,curtst, ". FAIL: allocated(t_allocarr2)"
  else 
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
  end if
  curtst = curtst + 1

  allocate(t_allocarr(2))
  result(curtst) = allocated(t_allocarr)
  if( result(curtst) ) then
!     print *,curtst, ". PASS: allocated(t_allocarr)"
  else 
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
  end if
  curtst = curtst + 1

  result(curtst) = allocated(t_allocarr(1)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr(1)%alloc_iary)"
  else 
!     print *, curtst, ". PASS: allocated(t_allocarr(1)%alloc_iary)"
  end if
  curtst = curtst + 1

  t_allocarr2 = t_allocarr
  result(curtst) = allocated(t_allocarr2)
  if( result(curtst) ) then
!     print *,curtst, ". PASS: allocated(t_allocarr2)"
  else 
!     print *,curtst, ". FAIL: allocated(t_allocarr2)"
  end if
  curtst = curtst + 1

  allocate( t_allocarr(1)%alloc_iary(2) )
  call init_t(t_allocarr(1), 1,66)
  result(curtst) = allocated(t_allocarr(1)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr(1)%alloc_iary))"
  else 
!     print *, curtst, ". FAIL: allocated(t_allocarr(1)%alloc_iary)"
  end if
  curtst = curtst + 1

! print *,"TEST 10"

  result(curtst) = allocated(t_allocarr(2)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr(2)%alloc_iary)"
  else
!     print *, curtst, ". PASS: allocated(t_allocarr(2)%alloc_iary))"
  end if
  curtst = curtst + 1
  
  t_allocarr2 = t_allocarr
  result(curtst) = allocated(t_allocarr2(1)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr2(1)%alloc_iary(2))"
  else 
!     print *, curtst, ". FAIL: allocated(t_allocarr2(1)%alloc_iary(2))"
  end if
  curtst = curtst + 1

  result(curtst) = allocated(t_allocarr2(2)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr2(2)%alloc_iary(2))"
  else 
!     print *, curtst, ". PASS: allocated(t_allocarr2(2)%alloc_iary(2))"
  end if
  curtst = curtst + 1

  deallocate( t_allocarr(1)%alloc_iary)
  result(curtst) = allocated(t_allocarr(1)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr(1)%alloc_iary)"
  else
!     print *, curtst, ". PASS: allocated(t_allocarr(1)%alloc_iary)"
  end if
  curtst = curtst + 1

  allocate( t_allocarr(2)%alloc_iary(2) )
  result(curtst) = allocated(t_allocarr(2)%alloc_iary)
  if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr(2)%alloc_iary)"
  else
!     print *, curtst, ". FAIL: allocated(t_allocarr(2)%alloc_iary)"
  end if
  curtst = curtst + 1

  deallocate(t_allocarr);
  allocate(t_allocarr(3))
  result(curtst) = allocated(t_allocarr)
  if( result(curtst) ) then
!     print *,curtst, ". PASS: allocated(t_allocarr)"
  else 
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
  end if
  curtst = curtst + 1
 
  result(curtst) = allocated(t_allocarr(1)%alloc_iary)
  if( result(curtst) ) then
!    print *,curtst, ". FAIL: alloc(1)%alloc_iary"
  else
!    print *,curtst, ". PASS: alloc(1)%alloc_iary"
  endif
  curtst = curtst + 1

  if(  allocated(t_allocarr(2)%alloc_iary) ) then
!    print *,curtst, ". FAIL: t_allocarr(2)%alloc_iary"
  else
!    print *,curtst, ". PASS: t_allocarr(2)%alloc_iary"
  endif
  curtst = curtst + 1

  deallocate(t_allocarr)

  t_allocarr2 = t_allocarr
  result(curtst) = allocated(t_allocarr2)
  if( result(curtst) ) then
!    print *, curtst, ". FAIL: allocated(t_allocarr2)"
  else
!    print *, curtst, ". PASS: allocated(t_allocarr2)"
  end if
  curtst = curtst + 1

  allocate(t_allocarr(2))
  t_allocarr = (/ t(1,NULL()), t(2,NULL()) /)
  t_allocarr(1) = t(1,NULL())
  t_allocarr(1)%i = 1
  t_allocarr(1)%alloc_iary = NULL()
  t_allocarr(2)%i = 2
  t_allocarr(2)%alloc_iary = NULL()

  t_allocarr2 = t_allocarr
  result(curtst) = allocated(t_allocarr2)
  if( result(curtst) ) then
!    print *, curtst, ". PASS: allocated(t_allocarr2)"
  else
!    print *, curtst, ". FAIL: allocated(t_allocarr2)"
  end if
   curtst = curtst + 1

! print *,"TEST 20"

   result(curtst) = allocated(l_inst%alloc_t_inst)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(l_inst%alloc_t_inst)"
   else
!     print *, curtst, ". PASS: allocated(l_inst%alloc_t_inst)"
   end if
   curtst = curtst + 1
  
   result(curtst) = allocated(ialloc)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(ialloc)"
   else
!     print *, curtst, ". PASS: allocated(ialloc)"
   end if
   curtst = curtst + 1
  
   deallocate(t_allocarr)
   result(curtst) = allocated(t_allocarr)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
   else
!     print *, curtst, ". PASS: allocated(t_allocarr)"
   end if
   curtst = curtst + 1
  
   call s1(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F F F"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   call s2(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F F F"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   call s3(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, "a. EXPECT: F F F"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   allocate(t_allocarr(2))
   result(curtst) = allocated(t_allocarr)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr)"
   else
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
   end if
   curtst = curtst + 1

! print *,"TEST 33"

   call s1(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F F T"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   call s2(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F F F"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 1

   allocate(l_iary(2))
   allocate(ialloc)
   allocate(t_allocarr(2))
     
   result(curtst) = allocated(l_iary)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(l_iary)"
   else
!     print *, curtst, ". PASS: allocated(l_iary)"
   end if
   curtst = curtst + 1
     
   result(curtst) = allocated(ialloc)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(ialloc)"
   else
!     print *, curtst, ". PASS: allocated(ialloc)"
   end if
   curtst = curtst + 3

! print *,"TEST 41"
 
   call s1(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: T T T"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   call s2(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F T F"
!   print *,result(curtst:curtst+2)
   curtst = curtst + 3

   call s3(ialloc, l_iary,t_allocarr,result(curtst:curtst+2))
!   print *, curtst, ". EXPECT: F F F"
!   print *,result(curtst:curtst+2)

! functions

   deallocate(t_allocarr2) 
   curtst = curtst + 1
   result(curtst) = allocated(ialloc)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(ialloc)"
   else
!     print *, curtst, ". PASS: allocated(ialloc)"
   end if
   curtst = curtst + 1
  
   result(curtst) = allocated(l_iary)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(l_iary)"
   else
!     print *, curtst, ". PASS: allocated(l_iary)"
     end if
   curtst = curtst + 1

! print *,"TEST 50"

   result(curtst) = allocated(t_allocarr)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
   else
!     print *, curtst, ". PASS: allocated(t_allocarr)"
   end if
   curtst = curtst + 1
  
   result(curtst) = allocated(t_allocarr2)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: allocated(t_allocarr2)"
   else
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
   end if
   curtst = curtst + 1
  
   ialloc2 = f1(ialloc,l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3
   deallocate(ialloc2);

   allocate(ialloc2)
   allocate(l_iary2(2))
   allocate(t_allocarr2(2))

   ialloc2 = f1(ialloc,l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(ialloc2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(ialloc2)"
   else
!     print *, curtst, ". FAIL: allocated(ialloc2)"
   end if
   curtst = curtst + 1
  
! print *,"TEST 59"

   l_iary2 = f2(ialloc,l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(l_iary2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
   else
!     print *, curtst, ". FAIL: allocated(t_allocarr2)"
   end if
   curtst = curtst + 1
  
   t_allocarr2 = f3(ialloc,l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(t_allocarr2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
   else
!     print *, curtst, ". FAIL allocated(t_allocarr2)"
   end if
   curtst = curtst + 1

   allocate(t_allocarr(2))
   result(curtst) = allocated(t_allocarr)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr)"
   else
!     print *, curtst, ". FAIL: allocated(t_allocarr)"
   end if

   deallocate(ialloc2)
   deallocate(l_iary2)
   deallocate(t_allocarr2)

   ialloc2 = f1(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F T"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

! print*,"TEST 70"

   result(curtst) = allocated(ialloc2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(ialloc2)"
   else
!     print *, curtst, ". FAIL allocated(ialloc2)"
   end if
   curtst = curtst + 1

   l_iary2 = f2(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(l_iary2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(l_iary2)"
   else
!     print *, curtst, ". FAIL: allocated(l_iary2)"
   end if
   curtst = curtst + 1

   allocate(t_allocarr(3))
   t_allocarr2 = f3(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F F F T" 
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(t_allocarr2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
   else
!     print *, curtst, ". FAIL: allocated(t_allocarr2)"
   end if
   curtst = curtst + 1

   allocate(l_iary(2))
   allocate(ialloc)
     
   ialloc2 = f1(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F T T T"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

! print *,"TEST 82"

   result(curtst) = allocated(ialloc2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(ialloc2)"
   else
!     print *, curtst, ". FAIL: allocated(ialloc2)"
   end if
   curtst = curtst + 1

   l_iary2 = f2(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F T T F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(l_iary2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(l_iary2)"
   else
!     print *, curtst, ". FAIL: allocated(l_iary2)"
   end if
   curtst = curtst + 1

   result(curtst) =  addrRetVal .eq. loc(l_iary2)
   if( result(curtst) ) then
!     print *, curtst, ". FAIL: addrRetVal .eq. loc(l_iary2)"
   else
!     print *, curtst, ". PASS: l_iary2 reallocated"
   end if

   allocate(ialloc)
   t_allocarr2  = f3(ialloc, l_iary,t_allocarr,result(curtst:curtst+3))
!   print *, curtst, ". EXPECT: F T F F"
!   print *,result(curtst:curtst+3)
   curtst = curtst + 3

   result(curtst) = allocated(t_allocarr2)
   if( result(curtst) ) then
!     print *, curtst, ". PASS: allocated(t_allocarr2)"
   else
!     print *, curtst, ". FAIL: allocated(t_allocarr2)"
   end if
  curtst = curtst + 1
 
 call check(result,expect, N)

end program
