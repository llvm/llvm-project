!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

program p
 use check_mod
 use iso_c_binding

  integer, parameter :: N =17
  integer :: rslt(N)
  integer :: expct(N) = (/1,1,2,4,8,1,2,4,8,8,16,0,0,4,8,16,64/)

  character(1, C_CHAR) :: c_c

  integer(C_SIGNED_CHAR) :: c_sc
  integer(C_SHORT) :: c_s
  integer(C_INT) :: c_i
  integer(C_LONG_LONG) :: c_ll

  integer(C_INT8_T) :: i1
  integer(C_INT16_T) :: i2
  integer(C_INT32_T) :: i4
  integer(C_INT64_T) :: i8

  integer(c_size_t) :: sz

  real(C_FLOAT) :: f
  real(C_DOUBLE) :: d

  complex(C_FLOAT_COMPLEX) :: f_cplx
  complex(C_DOUBLE_COMPLEX) :: d_cplx

  type  :: t 
    integer ::  i
  endtype

  type, bind(C)  :: t_c
    integer(c_int) ::  i
  endtype

  type :: th2_c
    integer(c_int) ::  i
  endtype

  type  :: t1
    integer ::  i
    real*8 :: d
  endtype

  type, bind(C)  :: t1_c
    integer(c_int64_t) ::  i
    real(c_double) :: d
  endtype

  type(t), target :: t_inst
  type(t) :: t_arr2(2)
  class(t), pointer :: t_class_inst

  type(t_c), target :: t_c_inst
  type(t_c) :: t_c_arr2(2)

  type(t1), target :: t1_inst
  type(t1) :: t1_arr2(2)
  class(t1), pointer :: t1_class_inst

  type(t1_c), target :: t1_c_inst
  type(t1_c) :: t1_c_arr4(4)

  type(c_ptr) :: cptr
  type(c_funptr) :: cfuncptr

  expct(12) = sizeof(cptr)
  expct(13) = sizeof(cfuncptr)

  rslt(1) = C_sizeof(c_c)
!  print *, "C_sizeof(c_c):", rslt(1)

  rslt(2) = C_sizeof(c_sc)
!  print *, "C_sizeof(c_sc):", rslt(2)

  rslt(3) = C_sizeof(c_s)
!  print *, "C_sizeof(c_s):", rslt(3)

  rslt(4) = C_sizeof(c_i)
!  print *, "C_sizeof(c_i):", rslt(4)

  rslt(5) = C_sizeof(c_ll)
!  print *, "C_sizeof(c_ll):", rslt(5)

  rslt(6) = C_sizeof(i1)
!  print *, "C_sizeof(i1):", rslt(6)

  rslt(7) = C_sizeof(i2)
!  print *, "C_sizeof(i2):", rslt(7)

  rslt(8) = C_sizeof(i4)
!  print *, "C_sizeof(i4):", rslt(8)

  rslt(9) = C_sizeof(i8)
!  print *, "C_sizeof(i8):", rslt(9)

  rslt(10) = C_sizeof(f_cplx)
!  print *, "C_sizeof(f_cplx):", rslt(10)

  rslt(11) = C_sizeof(d_cplx)
!  print *, "C_sizeof(d_cplx):", rslt(11)

  rslt(12) = C_sizeof(cptr)
!  print *,"C_sizeof(cptr):", rslt(12)

  rslt(13) = C_sizeof(cfuncptr)
!  print *,"C_sizeof(cfuncptr):", rslt(13)

!!  rslt() = C_sizeof(t_inst)		! should generate an error
!!  print *,"C_sizeof(t_inst):", rslt()

  rslt(14) = C_sizeof(t_c_inst)
!  print *,"C_sizeof(t_c_inst):", rslt(14)

  rslt(15) = C_sizeof(t_c_arr2)
!  print *,"C_sizeof(t_c_arr2):", rslt(15)

!!  rslt() = C_sizeof(t1_inst)		! should generate an error
!!  print *,"C_sizeof(t1_inst):", rslt()

  rslt(16) = C_sizeof(t1_c_inst)
!  print *,"C_sizeof(t1_c_inst):", rslt(16)

  rslt(17) = C_sizeof(t1_c_arr4)
!  print *,"C_sizeof(t1_c_arr4):", rslt(17)

!  print *,"expct:"
!  print *,expct
!  print *,"rslt:"
!  print *,rslt

  call check(rslt, expct, N);

end program
