! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


! directives.h -- contains preprocessor directives for F90 rte files

#include "mmul_dir.h"

subroutine ftn_transpose_cmplx8( ta, a, lda, alpha, buffer, bufrows, bufcols )
  implicit none
  integer*8 lda
  integer :: bufrows, bufcols
  integer i, j, ndx, ndxsave
  complex*8 :: a( lda, * ), alpha
  complex*8 :: buffer(bufrows * bufcols)
  integer :: ta

  !
  !   The plan here is to copy the matrix a to the buffer, or at least a 
  !   portion of it, such that the matrix  (really a buffer) is in proper 
  !   order for successive access. Some number of columns of a will be 
  !   dispersed to buffer to minimize page faults.
  !   The calling function can manage the buffer for both L1 and L2 cache
  !   utilization. bufcols defines the number of values taken from L1 cache
  !   for each dot product. bufrows * bufcols defines how much L2 cache is 
  !   used.
  !
  !   We may want to change this to be able to handle multiple sections of L1
  !   cache usage such as giving an additional parameter, say, nbufrows
  !   which would essentially copy more of the matrix a to the buffer using
  !   an additional loop
  
  !
  !   What do the parameters mean?
  !   buffer: buffer array
  !   a: matrix to be transposed
  !   bufcols: number of rows in matrix a to transpose
  !   bufrowss: number of cols in matrix a to transpose
  !   lda: number of rows in matrix a
  !   Note that we don't care what the dimensions of a are. We assume that the
  !   calling function has done this correctly
  ! 
  ndxsave = 1
  if( alpha .eq. 1.0 )then
     if( ta .eq. 2 )then ! conjugate the data on transfer to buffer
        do j = 1, bufrows
           ndx = ndxsave
           do i = 1, bufcols
              buffer( ndx ) = conjg( a( i, j ) )
              ndx = ndx + bufrows
           enddo
           ndxsave = ndxsave + 1
        enddo
     else
        do j = 1, bufrows
           ndx = ndxsave
           do i = 1, bufcols
              buffer( ndx ) = a( i, j ) 
              ndx = ndx + bufrows
           enddo
           ndxsave = ndxsave + 1
        enddo
     endif
  else
     if( ta .eq. 2 )then ! conjugate the data on transfer to buffer
        do j = 1, bufrows
           ndx = ndxsave
           do i = 1, bufcols
              buffer( ndx ) = alpha * conjg( a( i, j ) )
              ndx = ndx + bufrows
           enddo
           ndxsave = ndxsave + 1
        enddo
     else
        do j = 1, bufrows
           ndx = ndxsave
           do i = 1, bufcols
              buffer( ndx ) = alpha * a( i, j ) 
              ndx = ndx + bufrows
           enddo
           ndxsave = ndxsave + 1
        enddo
     endif
  endif
  !      write( *, * ) ( a(1, j ), j = 1, bufcols )
  !      write( *, * )( buffer( i ), i = 1, bufrows * bufcols )
  return
end subroutine ftn_transpose_cmplx8
