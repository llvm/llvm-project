! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


! directives.h -- contains preprocessor directives for F90 rte files

#include "mmul_dir.h"

subroutine ftn_mnaxtb_cmplx16( mra, ncb, kab, alpha, a, lda, b, ldb, beta, &
     & c, ldc )
  implicit none
#include "pgf90_mmul_cmplx16.h"
  
  ! 
  ! The main idea here is that the bufrows will define the usage of the
  ! L1 cache. We reference the same column or columns multiply while 
  ! accessing multiple partial rows of matrix a transposed in the buffer.
  
  !           Remember that everything is buffer centric
  !
  !
  !           <- bufca(1)>< (2)>                         <-bufcb->      
  !               i = 1, m                                j = 1, n
  !                rowsa                                  colsb
  !                  ar --->                                bc --->
  !      ^    +----------+------+   ^              +----------+----+  ^
  !      |    |          x      |   |              |   b      x    |  |
  !      |    |          x      |   |              |   u      x    |  |
  !  bufr(1)  |  A**T    x      | rowchunks=2      |   f  a   x c  |  |
  !      |    |          x      |   |              |   f      x    |  |
  !      |    | buffera  x      |   |              |   e      x    | kab = 1, k
  !      |    |          x      |   |          br  |   r      x    |  |
  !      |    |    I     x III  |   |          |   |   b      x    |  |
  !      v    +xxxxxxxxxxxxxxxxx+   |          |   +xxxxxxxxxx+xxxx|  |
  !      ^    |          x      |   |          v   |          x    |  |
  !      |    |   II     x IV   |   |              |   B  b   x d  |  |
  !   bufr(2) |          x      |   |              |          x    |  |
  !      |    |          x      |   |              |          x    |  |
  !      V    +----------+------+   V              +----------+----+  V
  !            <--colchunks=2-->
  !     x's mark buffer boudaries on the transposed matrices
  !     For this case, bufca(1) = bufcols, bufr(1) = bufrows
  !
  !    Algorimically, we perform dot products of (I,a), (III,a), (II,b)
  !    and (IV,b). The partial dot products of (I,a) are added to those 
  !    of (II,b) and those of (III,a) are added to those of (IV,b)
  !
  ! Iterations over the "chunks" are buffer based 
  ! while iterations over i and j are matrix based and keep track of where
  ! we are in the larger scheme of things
  ! Iterations over i and j are bounded by buffer dimensions
  !
  colsa = kab
  rowsb = kab
  rowsa = mra
  colsb = ncb
  if (colsa * rowsa * colsb < min_blocked_mult) then
    if( beta .eq. 0.0 ) then
      do j = 1, colsb
         do i = 1, rowsa
            temprr0 = 0.0
            tempri0 = 0.0
            tempir0 = 0.0
            tempii0 = 0.0
            do k = 1, colsa
                temprr0 = temprr0 + real(alpha) * real(a(i, k)) * real(b(j, k)) - aimag(alpha) * aimag(a(i, k)) * real(b(j, k))
            enddo
            do k = 1, colsa
                tempii0 = tempii0 + real(alpha) * aimag(a(i, k)) * aimag(b(j, k)) + aimag(alpha) * real(a(i, k)) * aimag(b(j, k))
            enddo
            do k = 1, colsa
                tempir0 = tempir0 + real(alpha) * real(a(i, k)) * aimag(b(j, k)) - aimag(alpha) * aimag(a(i, k)) * aimag(b(j, k))
            enddo
            do k = 1, colsa
                tempri0 = tempri0 + real(alpha) * aimag(a(i, k)) * real(b(j, k)) + aimag(alpha) * real(a(i, k)) * real(b(j, k))
            enddo
            c(i, j) = dcmplx((temprr0 - tempii0), (tempri0 + tempir0))
         enddo
      enddo
    else
      do j = 1, colsb
         do i = 1, rowsa
            temprr0 = 0.0
            tempri0 = 0.0
            tempir0 = 0.0
            tempii0 = 0.0
            do k = 1, colsa
                temprr0 = temprr0 + real(alpha) * real(a(i, k)) * real(b(j, k)) - aimag(alpha) * aimag(a(i, k)) * real(b(j, k))
            enddo
            do k = 1, colsa
                tempii0 = tempii0 + real(alpha) * aimag(a(i, k)) * aimag(b(j, k)) + aimag(alpha) * real(a(i, k)) * aimag(b(j, k))
            enddo
            do k = 1, colsa
                tempir0 = tempir0 + real(alpha) * real(a(i, k)) * aimag(b(j, k)) - aimag(alpha) * aimag(a(i, k)) * aimag(b(j, k))
            enddo
            do k = 1, colsa
                tempri0 = tempri0 + real(alpha) * aimag(a(i, k)) * real(b(j, k)) + aimag(alpha) * real(a(i, k)) * real(b(j, k))
            enddo
            
            c(i, j) = beta * c(i, j) + dcmplx((temprr0 - tempii0), (tempri0 + tempir0))
         enddo
      enddo
    endif
  else
    allocate( buffera( bufrows * bufcols ) )
    allocate( bufferb( bufrows * bufcols ) )

    ! for algoritmic purposes, kab is the number of columns in matrix b, which
    ! is also the number of columns in matrix a.


    bufr = min( bufrows, colsa )
    bufr_sav = bufr
    bufca = min( bufcols, rowsa )
    bufca_sav = bufca
    bufcb = min( bufcols, colsb )
    bufcb_sav = bufcb
    ar_sav = 1
    ac_sav = 1
    bc = 1
    br = 1
    ! both rowchunks and colchunks are buffer centric
    rowchunks = ( colsa + bufr - 1 )/bufr 
    colachunks = ( rowsa + bufca - 1 )/bufca 
    colbchunks = ( colsb + bufcb - 1 )/bufcb
    ! these are for loop unrolling
    colsb_chunk = 4
    
    do rowchunk = 1, rowchunks
       bufcb = bufcb_sav
       do colbchunk = 1, colbchunks
          bufcb = min( bufcb_sav, colsb - bc + 1 )
          bufr = min( bufr_sav, rowsb - br + 1 )
          call ftn_transpose_cmplx16( tb, b( bc, br ), ldb, alpha, bufferb, &
               & bufr, bufcb )
          !       ar = ar_sav
          !       ac = 1
          ar = 1
          ac = ac_sav
          do colachunk = 1, colachunks
             if( br .eq. 1 )then
                ! Note: alpha is 1.0 for matrix a to avoid multiplying by 
                ! alpha * alpha
                bufca = min( bufca_sav, rowsa - ar + 1 )
                call ftn_transpose_cmplx16( ta, a( ar, ac ), lda, one, buffera, &
                     & bufr, bufca )
                ndxb0 = 0
                ndxb1 = bufr
                ndxb2 = ndxb1 + bufr
                ndxb3 = ndxb2 + bufr
                colsb_chunks = bufcb/colsb_chunk
                colsb_end = bc + colsb_chunks * colsb_chunk - 1 
                colsb_strt = colsb_end + 1
                jend = bc + bufcb - 1
                j = bc
                if( beta .eq. 0.0 ) then
                   do jb = 1, colsb_chunks
                      ndxa = 0
                      do i = ar, ar + bufca - 1
                         temp0 = 0.0
                         temp1 = 0.0
                         temp2 = 0.0
                         temp3 = 0.0
                         do k = 1, bufr
                            bufatemp = buffera( ndxa + k )
                            temp0 = temp0 + bufferb( ndxb0 + k ) * bufatemp
                            temp1 = temp1 + bufferb( ndxb1 + k ) * bufatemp
                            temp2 = temp2 + bufferb( ndxb2 + k ) * bufatemp
                            temp3 = temp3 + bufferb( ndxb3 + k ) * bufatemp
                         enddo
                         c( i, j )     = temp0
                         c( i, j + 1 ) = temp1
                         c( i, j + 2 ) = temp2
                         c( i, j + 3 ) = temp3
                         ndxa = ndxa + bufr
                      enddo
                      ndxa = 0
                      ndxb0 = ndxb0 + bufr * colsb_chunk
                      ndxb1 = ndxb1 + bufr * colsb_chunk
                      ndxb2 = ndxb2 + bufr * colsb_chunk
                      ndxb3 = ndxb3 + bufr * colsb_chunk
                      j = j + 4
                   enddo
                   ndxb = bufr * colsb_chunks * colsb_chunk
                   do j = colsb_strt, jend
                      ndxa = 0
                      do i = ar, ar + bufca - 1
                         temp = 0.0
                         do k = 1, bufr
                            temp = temp + bufferb( ndxb + k ) * &
                                 & buffera( ndxa + k )
                         enddo
                         c( i, j ) = temp
                         ndxa = ndxa + bufr
                      enddo
                      ndxb = ndxb + bufr
                   enddo
                   !             ac = ac + bufca
                   ar = ar + bufca
                   !           print *, "ac: ", ac
                else
                   do jb = 1, colsb_chunks
                      ndxa = 0
                      do i = ar, ar + bufca - 1
                         temp0 = 0.0
                         temp1 = 0.0
                         temp2 = 0.0
                         temp3 = 0.0
                         do k = 1, bufr
                            bufatemp = buffera( ndxa + k )
                            temp0 = temp0 + bufferb( ndxb0 + k ) * bufatemp
                            temp1 = temp1 + bufferb( ndxb1 + k ) * bufatemp
                            temp2 = temp2 + bufferb( ndxb2 + k ) * bufatemp
                            temp3 = temp3 + bufferb( ndxb3 + k ) * bufatemp
                         enddo
                         c( i, j )     = beta * c( i, j )     + temp0
                         c( i, j + 1 ) = beta * c( i, j + 1 ) + temp1
                         c( i, j + 2 ) = beta * c( i, j + 2 ) + temp2
                         c( i, j + 3 ) = beta * c( i, j + 3 ) + temp3
                         ndxa = ndxa + bufr
                      enddo
                      ndxa = 0
                      ndxb0 = ndxb0 + bufr * colsb_chunk
                      ndxb1 = ndxb1 + bufr * colsb_chunk
                      ndxb2 = ndxb2 + bufr * colsb_chunk
                      ndxb3 = ndxb3 + bufr * colsb_chunk
                      j = j + 4
                   enddo
                   ndxb = bufr * colsb_chunks * colsb_chunk
                   do j = colsb_strt, jend
                      ndxa = 0
                      do i = ar, ar + bufca - 1
                         temp = 0.0
                         do k = 1, bufr
                            temp = temp + bufferb( ndxb + k ) * &
                                 & buffera( ndxa + k )
                         enddo
                         c( i, j ) = beta * c( i, j ) + temp
                         ndxa = ndxa + bufr
                      enddo
                      ndxb = ndxb + bufr
                   enddo
                   !             ac = ac + bufca
                   ar = ar + bufca
                   !           print *, "ac: ", ac
                endif
             else
                bufca = min( bufca_sav, rowsa - ar + 1 )
                call ftn_transpose_cmplx16( ta, a( ar, ac ), lda, one  , buffera, &
                     & bufr, bufca )
                ndxb0 = 0
                ndxb1 = bufr
                ndxb2 = ndxb1 + bufr
                ndxb3 = ndxb2 + bufr
                colsb_chunks = bufcb/colsb_chunk
                colsb_end = bc + colsb_chunks * colsb_chunk - 1 
                colsb_strt = colsb_end + 1
                jend = bc + bufcb - 1
                j = bc
                do jb = 1, colsb_chunks
                   ndxa = 0
                   do i = ar, ar + bufca - 1
                      temp0 = 0.0
                      temp1 = 0.0
                      temp2 = 0.0
                      temp3 = 0.0
                      do k = 1, bufr
                         bufatemp = buffera( ndxa + k )
                         temp0 = temp0 + bufferb( ndxb0 + k ) * bufatemp
                         temp1 = temp1 + bufferb( ndxb1 + k ) * bufatemp
                         temp2 = temp2 + bufferb( ndxb2 + k ) * bufatemp
                         temp3 = temp3 + bufferb( ndxb3 + k ) * bufatemp
                      enddo
                      c( i, j )     = c( i, j )     + temp0
                      c( i, j + 1 ) = c( i, j + 1 ) + temp1
                      c( i, j + 2 ) = c( i, j + 2 ) + temp2
                      c( i, j + 3 ) = c( i, j + 3 ) + temp3
                      ndxa = ndxa + bufr
                   enddo
                   ndxa = 0
                   ndxb0 = ndxb0 + bufr * colsb_chunk
                   ndxb1 = ndxb1 + bufr * colsb_chunk
                   ndxb2 = ndxb2 + bufr * colsb_chunk
                   ndxb3 = ndxb3 + bufr * colsb_chunk
                   j = j + 4
                enddo
                ndxb = bufr * colsb_chunks * colsb_chunk
                do j = colsb_strt, jend
                   ndxa = 0
                   do i = ar, ar + bufca - 1
                      temp = 0.0
                      do k = 1, bufr
                         temp = temp + bufferb( ndxb + k ) * buffera( ndxa + k )
                      enddo
                      c( i, j ) = c( i, j ) + temp
                      ndxa = ndxa + bufr
                   enddo
                   ndxb = ndxb + bufr
                enddo
                ar = ar + bufca
             endif
          enddo
          
          bc = bc + bufcb
       enddo
       br = br + bufr
       ac_sav = ac_sav + bufr
       bc = 1
    enddo
    deallocate( buffera )
    deallocate( bufferb )
  endif
  return
end subroutine ftn_mnaxtb_cmplx16
