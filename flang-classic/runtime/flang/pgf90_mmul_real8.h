! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

  !
  ! Global variables
  !
  integer*8 :: mra, ncb, kab, lda, ldb, ldc
  real*8, dimension( lda, * )::a
  real*8, dimension( ldb, * )::b
  real*8, dimension( ldc, * )::c
  real*8 :: alpha, beta, one = 1.0

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
  real*8   :: temp, temp0, temp1, temp2, temp3 
  real*8   :: bufatemp, bufbtemp
  real*8   :: time_start, time_end, ttime, all_time

  integer, parameter :: bufrows = 512, bufcols = 8192
!  integer, parameter :: bufrows = 2, bufcols = 3
!  real*8, dimension( bufrows * bufcols ) :: buffera, bufferb
  real*8, allocatable, dimension(:) :: buffera, bufferb

  !Minimun number of multiplications needed to activate the blocked optimization.
#ifdef TARGET_X8664
  integer, parameter :: min_blocked_mult = 5000
#elif TARGET_LINUX_POWER
  integer, parameter :: min_blocked_mult = 10000
#else
  #warning untuned matrix multiplication parameter
  integer, parameter :: min_blocked_mult = 5000 
#endif

