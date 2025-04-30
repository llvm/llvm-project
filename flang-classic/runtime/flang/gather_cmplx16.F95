! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


! directives.h -- contains preprocessor directives for F90 rte files

#include "mmul_dir.h"

subroutine ftn_gather_cmplx16( ta, a, lda, alpha, buffer, bufrows, bufcols )
  implicit none

  integer*8 lda
  complex*16 :: a( lda,* ), alpha
  integer :: bufrows, bufcols
  integer :: i, j, ndx, ndxsave
  complex*16 :: buffer(bufrows * bufcols)
  integer   :: ta
  !
  ! This routine gathers the matrix into l1 chunks. The purpose is much as it
  ! is for the transpose case, and works much like transpose_real8()
  !
  !   What do the parameters mean?
  !   buffer: buffer array
  !   a: matrix to be gathered
  !   bufcols: number of rows in matrix a to gather
  !   bufrowss: number of cols in matrix a to gather
  !   lda: number of rows in matrix a
  !   Note that we don't care what the dimensions of a are. We assume that the
  !   calling function has done this correctly
  !
  
  ndx = 0
  if( ta .eq. 2 )then ! conjugate the data
     if( alpha .eq. ( 1.0, 0.0 ) ) then
        do j = 1, bufcols
           do i = 1, bufrows
              buffer( ndx + i ) = conjg(  a( i, j ) )
           enddo
           ndx = ndx + bufrows
        enddo
     else
        do j = 1, bufcols
           do i = 1, bufrows
              buffer( ndx + i ) = alpha * conjg(  a( i, j ) )
           enddo
           ndx = ndx + bufrows
        enddo
     endif
  else
     if( alpha .eq. ( 1.0, 0.0 ) ) then
        do j = 1, bufcols
           do i = 1, bufrows
              buffer( ndx + i ) = a( i, j ) 
           enddo
           ndx = ndx + bufrows
        enddo
     else
        do j = 1, bufcols
           do i = 1, bufrows
              buffer( ndx + i ) = alpha * a( i, j )
           enddo
           ndx = ndx + bufrows
        enddo
     endif
  endif
     !      write( *, * ) ( a(1, j ), j = 1, bufcols )
  !      write( *, * )( buffer( i ), i = 1, bufrows * bufcols )
  return
end subroutine ftn_gather_cmplx16
