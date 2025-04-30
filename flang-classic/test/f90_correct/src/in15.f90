!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test f2008 asinh intrinsic

program p 
 use ISO_C_BINDING
 use check_mod

  interface
    subroutine get_expected_f( src1, expct, n ) bind(C)
     use ISO_C_BINDING
      type(C_PTR), value :: src1
      type(C_PTR), value :: expct
      integer(C_INT), value :: n
    end subroutine
    
    subroutine get_expected_d( src1, expct, n ) bind(C)
     use ISO_C_BINDING
      type(C_PTR), value :: src1
      type(C_PTR), value :: expct
      integer(C_INT), value :: n
    end subroutine
  end interface
    

  integer, parameter :: N=20
  real*4, target,  dimension(N) :: r_src1
  real*4, target,  dimension(N) :: r_rslt
  real*4, target,  dimension(N) :: r_expct
  real*8 :: valuer
  
  real*8, target,  dimension(N) :: d_src1
  real*8, target,  dimension(N) :: d_rslt
  real*8, target,  dimension(N) :: d_expct
  real*8 :: value8
  
  valuer = -5.5
  valued = -5.5_8
  do i =  1,N 
    r_src1(i) = valuer
    d_src1(i) = valued
    valuer = valuer + .5
    valued = valued + .5_8
  enddo

  r_rslt = asinh(r_src1)
  d_rslt = asinh(d_src1)

  call get_expected_f(C_LOC(r_src1), C_LOC(r_expct), N)
  call get_expected_d(C_LOC(d_src1), C_LOC(d_expct), N)

  call checkr4( r_rslt, r_expct, N, rtoler=0.0000003)
  call checkr8( d_rslt, d_expct, N, rtoler=0.0000003_8)

!   print *, "r_expct:" 
!   print *, r_expct
!   print *, "r_rslt:" 
!   print *, r_rslt

!   print *, "d_expct:" 
!   print *, d_expct
!   print *, "d_rslt:" 
!   print *, d_rslt
end program 
