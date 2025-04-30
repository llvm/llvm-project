!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test f2008 bessel_yn transformational intrinsic

program p 
 use ISO_C_BINDING
 use check_mod

  interface
    subroutine get_expected_f( src1, expct, n1, n2, n ) bind(C)
     use ISO_C_BINDING
      real(C_FLOAT), value :: src1
      type(C_PTR), value :: expct
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      integer(C_INT), value :: n
    end subroutine
    
    subroutine get_expected_d( src1, expct, n1, n2, n ) bind(C)
     use ISO_C_BINDING
      real(C_DOUBLE), value :: src1
      type(C_PTR), value :: expct
      integer(C_INT), value :: n1
      integer(C_INT), value :: n2
      integer(C_INT), value :: n
    end subroutine
  end interface
    

  integer, parameter :: N1=0
  integer, parameter :: N2=4
  integer, parameter :: NBR_ORDERS = N2-N1+1
  integer, parameter :: TST_VALUES=5
  integer, parameter :: TSTS=NBR_ORDERS*TST_VALUES
  real*4, target,  dimension(TST_VALUES) :: r_src1
  real*4, target,  dimension(TSTS) :: r_rslt
  real*4, target,  dimension(TSTS) :: r_expct
  real*8 :: valuer
  
  real*8, target,  dimension(TST_VALUES) :: d_src1
  real*8, target,  dimension(TSTS) :: d_rslt
  real*8, target,  dimension(TSTS) :: d_expct
  real*8 :: valued
  integer :: order = 3
  
  valuer = .001
  valued = .001
  do i =  0,TST_VALUES-1
    r_src1(i+1) = valuer + 2*i
    d_src1(i+1) = valued + 2*i
  enddo

  do i=1, TST_VALUES
    r_rslt(1+(i-1)*NBR_ORDERS:) = bessel_yn(N1,N2, r_src1(i))
    d_rslt(1+(i-1)*NBR_ORDERS:) = bessel_yn(N1,N2, d_src1(i))

    call get_expected_f(r_src1(i), C_LOC(r_expct(1+(i-1)*NBR_ORDERS)), &
                        N1, N2, TSTS/NBR_ORDERS)
    call get_expected_d(d_src1(i), C_LOC(d_expct(1+(i-1)*NBR_ORDERS)), &
                         N1, N2, TSTS/NBR_ORDERS)
  enddo

  call checkr4( r_rslt, r_expct, TSTS, rtoler=0.0000003)
  call checkr8( d_rslt, d_expct, TSTS, rtoler=0.0000003_8)

!   print *, "r_expct:" 
!   print *, r_expct
!   print *, "r_rslt:" 
!   print *, r_rslt
!
!   print *, "d_expct:" 
!   print *, d_expct
!   print *, "d_rslt:" 
!   print *, d_rslt
end program 
