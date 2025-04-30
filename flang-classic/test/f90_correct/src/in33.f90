!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module dt

  type  :: t
    integer ::  i
    real, allocatable  ::f(:)
  endtype

  type, extends(t) :: t_e
    real*8 :: d
    real*8, allocatable :: darry(:)
  endtype

  type, extends(t_e) :: t_e_e
    real*8 :: d2
  endtype
end module

integer*8 function fc(c)
  character(*) :: c
  fc = storage_size(c)
end function

integer*8 function fcarr(c)
  character(*) :: c(:)
  fcarr = storage_size(c)
end function

integer*8 function fdt(dt_arg)
 use dt
  class(t) :: dt_arg

  fdt = storage_size(dt_arg)
end function

integer*8 function fdtarr(dtarr_arg)
 use dt
  class(t) :: dtarr_arg(:)

  fdtarr = storage_size(dtarr_arg)
end function


program p
 use dt
 use check_mod

  interface
   integer*8 function fc(c)
     character(len=*) :: c
   end function
 
   integer*8 function fcarr(c)
     character(len=*) :: c(:)
   end function
 
   integer*8 function fdt(dt_arg)
    use dt
     class(t) :: dt_arg
   end function
 
   integer*8 function fdtarr(dtarr_arg)
    use dt
     class(t) :: dtarr_arg(:)
   end function
  end interface

  type, extends(t_e_e) :: t_e_e_huge
    type(t_e_e) :: teearry(45)
  endtype

  integer, parameter :: N=54
  integer*8 :: rslt(N)
  integer*8 :: expct(N)
  
  integer :: iarr(4)
  integer, allocatable, dimension(:) :: alloc_iarr
  
  type(t), target :: t_inst
  type(t) :: t_arr2(2)
  type(t), target :: t_arr4(4)
  class(t), allocatable, target :: t_alloc(:)
  class(t), pointer :: t_ptr
  class(t), pointer :: t_ptr2
  class(t), pointer :: t_ptr3(:)

  type(t_e), target :: t_e_inst
  type(t_e) :: t_e_arr2(2)
  type(t_e), target :: t_e_arr4(4)
  class(t_e), allocatable, target :: t_e_alloc(:)
  class(t_e), pointer :: t_e_ptr
  class(t_e), pointer :: t_e_ptr2
  class(t_e), pointer :: t_e_ptr3(:)

  type(t_e_e), target :: t_e_e_inst
  type(t_e_e) :: t_e_e_arr2(2)
  type(t_e_e), target :: t_e_e_arr4(4)
  class(t_e_e), allocatable, target :: t_e_e_alloc(:)
  class(t_e_e), pointer :: t_e_e_ptr
  class(t_e_e), pointer :: t_e_e_ptr2
  class(t_e_e), pointer :: t_e_e_ptr3(:)

  type(t_e_e_huge), target :: teeh
  type(t_e_e_huge), target :: teeharr(2)

  character(len=5) :: c5
  character(len=5) :: c5arr(5)
  
  integer*1 :: sz1
  integer*2 :: sz2
  integer*4 :: sz4
  integer*8 :: sz8
  integer*8 :: t_sz
  integer*8 :: t_e_sz
  integer*8 :: t_e_e_sz
  integer*8 :: t_e_e_huge_sz

  real*4 :: f
  real*8 :: d

  integer :: i
  
  
  t_sz = (loc(t_arr2(2)) - loc(t_arr2(1))) * 8
!  print *, "loc(t_arr2(2)) - loc(t_arr2(1))", t_sz
  t_e_sz = (loc(t_e_arr2(2)) - loc(t_e_arr2(1))) * 8
!  print *, "loc(t_e_arr2(2)) - loc(t_e_arr2(1))", t_e_sz
  t_e_e_sz = (loc(t_e_e_arr2(2)) - loc(t_e_e_arr2(1))) * 8
!  print *, "loc(t_e_e_arr2(2)) - loc(t_e_e_arr2(1))", t_e_e_sz
  t_e_e_huge_sz = (loc(teeharr(2)) - loc(teeharr(1))) * 8
!  print *,"t_e_e_huge_sz", t_e_e_huge_sz

  expct(1) = bit_size(sz4)
  expct(2) = bit_size(iarr(1))

  do i=3,6
    expct(i) = 0
  end do

  do i=7,16
    expct(i) = t_sz
  end do

  do i=17,20
    expct(i) = 0
  end do

  do i=21,32
    expct(i) = t_e_sz
  end do

  do i=33,36
    expct(i) = 0
  end do

  do i=37,45
    expct(i) = t_e_e_sz
  end do

  sz1 = t_e_e_sz
!  print *,"sz1", sz1
  expct(46) = sz1
  sz2 = (loc(teeharr(2)) - loc(teeharr(1))) * 8
!  print *,"sz2", sz2
  expct(47) = sz2
  sz4 = (loc(teeharr(2)) - loc(teeharr(1))) * 8
!  print *,"sz2", sz2
  expct(48) = sz2

  do i = 49,52
      expct(i) = sizeof(c5) * 8
  end do

  expct(53) = sizeof(f) * 8
  expct(54) = sizeof(d) * 8

  rslt(1) = storage_size(sz4)
!  print *,"storage_size(sz4): ",rslt(1)
  rslt(2) = storage_size(iarr)
!  print *,"storage_size(iarr): ",rslt(2)
  
!  print *, "========================================================="
  rslt(3) = storage_size(t_alloc)
!  print *,"storage_size(t_alloc):", rslt(3)
  rslt(4) = storage_size(t_ptr)
!  print *,"storage_size(t_ptr):", rslt(4)
  rslt(5) = storage_size(t_ptr2)
!  print *,"storage_size(t_ptr1):", rslt(5)
  rslt(6) = storage_size(t_ptr3)
!  print *,"storage_size(t_ptr3):", rslt(6)
  rslt(7) = storage_size(t_inst)
!  print *,"storage_size(t_inst):", rslt(7)
  rslt(8) = storage_size(t_arr2)
!  print *,"storage_size(t_arr2):", rslt(8)
  rslt(9) = storage_size(t_arr4)
!  print *,"storage_size(t_arr4):", rslt(9)
  
  allocate(t_alloc(2))
  t_ptr=>t_inst
  t_ptr2=>t_alloc(1)
  t_ptr3=>t_arr4
  
  rslt(10) = storage_size(t_alloc)
!  print *,"storage_size(t_alloc):", rslt(10)
  rslt(11) = storage_size(t_ptr)
!  print *,"storage_size(t_ptr):", rslt(11)
  rslt(12) = storage_size(t_ptr2)
!  print *,"storage_size(t_ptr2):", rslt(12)
  rslt(13) = storage_size(t_ptr3)
!  print *,"storage_size(t_ptr3):", rslt(13)
  rslt(14) = fdt(t_inst)
!  print *," fdt(t_inst:", rslt(14)
  rslt(15) = fdt(t_ptr2)
!  print *," fdt(t_ptr2:", rslt(15)
  rslt(16) = fdtarr(t_ptr3)
!  print *," fdtarr(t_ptr3:", rslt(16)
  
!  print *, "========================================================="
  rslt(17) = storage_size(t_e_alloc)
!  print *,"storage_size(t_e_alloc):", rslt(17)
  rslt(18) = storage_size(t_e_ptr)
!  print *,"storage_size(t_e_ptr):", rslt(18)
  rslt(19) = storage_size(t_e_ptr2)
!  print *,"storage_size(t_e_ptr2):", rslt(19)
  rslt(20) = storage_size(t_e_ptr3)
!  print *,"storage_size(t_e_ptr3):", rslt(20)
  rslt(21) = storage_size(t_e_inst)
!  print *,"storage_size(t_e_inst):", rslt(21)
  rslt(22) = storage_size(t_e_arr2)
!  print *,"storage_size(t_e_arr2):", rslt(22)
  rslt(23) = storage_size(t_e_arr4)
!  print *,"storage_size(t_e_arr4):", rslt(23)
  
  allocate(t_e_alloc(2))
  t_e_ptr=>t_e_inst
  t_e_ptr2=>t_e_alloc(1)
  t_e_ptr3=>t_e_arr4
  t_ptr2=>t_e_ptr
  t_ptr3=>t_e_ptr3
  
  rslt(24) = storage_size(t_e_alloc)
!  print *,"storage_size(t_e_alloc):", rslt(24)
  rslt(25) = storage_size(t_e_ptr)
!  print *,"storage_size(t_e_ptr):", rslt(25)
  rslt(26) = storage_size(t_e_ptr2)
!  print *,"storage_size(t_e_ptr2):", rslt(26)
  rslt(27) = storage_size(t_e_ptr3)
!  print *,"storage_size(t_e_ptr3):", rslt(247
  rslt(28) = storage_size(t_ptr2)
!  print *,"storage_size(t_ptr2):", rslt(28)
  rslt(29) = storage_size(t_ptr3)
!  print *,"storage_size(t_ptr3):", rslt(29)
  rslt(30) = fdt(t_e_inst)
!  print *," fdt(t_e_inst):", rslt(30)
  rslt(31) = fdt(t_e_ptr)
!  print *," fdt(t_e_ptr):", rslt(31)
  rslt(32) = fdtarr(t_e_ptr3)
!  print *," fdtarr(t_e_ptr3):", rslt(32)

  
!  print *, "========================================================="
  rslt(33) = storage_size(t_e_e_alloc)
!  print *,"storage_size(t_e_e_alloc):", rslt(33)
  rslt(34) = storage_size(t_e_e_ptr)
!  print *,"storage_size(t_e_e_ptr):", rslt(34)
  rslt(35) = storage_size(t_e_e_ptr2)
!  print *,"storage_size(t_e_e_ptr2):", rslt(35)
  rslt(36) = storage_size(t_e_e_ptr3)
!  print *,"storage_size(t_e_e_ptr3):", rslt(36)
  rslt(37) = storage_size(t_e_e_inst)
!  print *,"storage_size(t_e_e_inst):", rslt(37)
  rslt(38) = storage_size(t_e_e_arr2)
!  print *,"storage_size(t_e_e_arr2):", rslt(38)
  rslt(39) = storage_size(t_e_e_arr4)
!  print *,"storage_size(t_e_e_arr4):", rslt(39)
  
  allocate(t_e_e_alloc(2))
  t_e_e_ptr=>t_e_e_inst
  t_e_e_ptr2=>t_e_e_alloc(1)
  t_e_e_ptr3=>t_e_e_arr4
  t_e_ptr2=>t_e_e_inst
  t_ptr3=>t_e_e_alloc
  
  rslt(40) = storage_size(t_e_e_alloc)
!  print *,"storage_size(t_e_e_alloc):", rslt(40)
  rslt(41) = storage_size(t_e_e_ptr)
!  print *,"storage_size(t_e_e_ptr):", rslt(41)
  rslt(42) = storage_size(t_e_e_ptr2)
!  print *,"storage_size(t_e_e_ptr2):", rslt(42)
  rslt(43) = storage_size(t_e_e_ptr3)
!  print *,"storage_size(t_e_e_ptr3):", rslt(43)
  rslt(44) = storage_size(t_e_ptr2)
!  print *,"storage_size(t_e_ptr2):", rslt(44)
  rslt(45) = storage_size(t_ptr3)
!  print *,"storage_size(t_ptr3):", rslt(45)
  
  sz1 = storage_size(t_ptr3,1)
!  print *,"storage_size(t_ptr3,1):", sz1
  rslt(46) = sz1
  
  t_e_ptr=>teeh
  sz2 = storage_size(t_e_ptr,2)
!  print *,"storage_size(t_e_ptr3,2):", sz2
  rslt(47) = sz2
  
  sz4 = storage_size(t_ptr3,4)
!  print *,"storage_size(t_ptr3,4):", sz4
  rslt(48) = sz2

!  sz8 = sizeof(c5)
!  print *,"sizeof(c5): ", sz8
!  sz8 = storage_size(c5)
!  print *,"storage_size(c5): ", sz8

  rslt(49) = storage_size(c5arr)
!  print *,"sizeof(c5arr): ", rslt(49)
  rslt(50) = storage_size(c5arr)
!  print *,"storage_size(c4arr): ", 50
  rslt(51) = fc(c5)
!   print *,"sz8 = fc(c5):", rslt(51)
  rslt(52) = fcarr(c5arr)
!   print *,"sz8 = fc(c5arr):", rslt(52)

    rslt(53) = storage_size(t_e_ptr%f)
!    print *,"storage_size(t_e_ptr%f):", rslt(53)

    rslt(54) = storage_size(t_e_ptr%darry)
!    print *,"storage_size(t_e_ptr%darry):", rslt(54)

!  print *,"expct: "
!  print *,expct

!  print *,"rslt: "
!  print *,rslt

  call checki8(rslt, expct, N);

end program p
