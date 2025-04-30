! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 

  !
  ! Global variables
  !
  integer*8 :: mra, ncb, kab, lda, ldb, ldc
  complex*16, dimension( lda, * )::a
  complex*16, dimension( ldb, * )::b
  complex*16, dimension( ldc, * )::c
  complex*16 :: alpha, beta, one = 1.0
    character*1 :: ca, cb
    !
    ! local variables
    !
  integer*8  :: colsa, rowsa, rowsb, colsb
  integer*8  :: i, j, jb, k, ak, bk, jend
  integer*8  :: ar, ar_sav,  ac, ac_sav, br, bc
  integer*8  :: ndxa, ndxasav 
  integer*8  :: ndxb, ndxbsav, ndxb0, ndxb1, ndxb2, ndxb3
  integer*8  :: colachunk, colachunks, colbchunk, colbchunks
  integer*8  :: rowchunk, rowchunks
  integer*8  :: colsb_chunk, colsb_chunks, colsb_strt, colsb_end
  integer*8  :: colsa_chunk, colsa_chunks, colsa_strt, colsa_end
  integer*8  :: bufr, bufr_sav, bufca, bufca_sav, bufcb, bufcb_sav
  integer  :: ta, tb
  complex*16   :: temp, temp0, temp1, temp2, temp3 
    real*8   :: temprr0, temprr1, temprr2, temprr3
    real*8   :: tempii0, tempii1, tempii2, tempii3
    real*8   :: tempri0, tempri1, tempri2, tempri3
    real*8   :: tempir0, tempir1, tempir2, tempir3
    complex*16   :: bufatemp, bufbtemp
    real*8    :: bufatempr, bufatempi, bufbtempr, bufbtempi
  real*8   :: time_start, time_end, ttime, all_time

  integer, parameter :: bufrows = 512, bufcols = 8192
!  integer, parameter :: bufrows = 2, bufcols = 3
!  complex*16, dimension( bufrows * bufcols ) :: buffera, bufferb
    complex*16, allocatable, dimension(:) :: buffera, bufferb
  
!Minimun number of multiplications needed to activate the blocked optimization.
#ifdef TARGET_X8664
  integer, parameter :: min_blocked_mult = 15000 
#elif TARGET_LINUX_POWER
  integer, parameter :: min_blocked_mult = 15000 !Complex calculations not vectorized on OpenPower.
#else
  #warning untuned matrix multiplication parameter
  integer, parameter :: min_blocked_mult = 15000 
#endif

#undef DCMPLX
#define DCMPLX(r,i) cmplx(r,i,kind=8)
