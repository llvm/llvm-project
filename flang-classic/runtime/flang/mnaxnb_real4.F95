! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

subroutine ftn_mnaxnb_real4( mra, ncb, kab, alpha, a, lda, b, ldb, beta, &
     & c, ldc )
  implicit none
#include "pgf90_mmul_real4.h"
  
  ! Everything herein is focused on how the transposition buffer maps
  ! to the matrix a. The size of the buffer is bufrows * bufcols
  ! Since once transposed data will be read from the buffer down the rows,
  ! bufrows corresponds to the columns of a while bufcols corresponds to
  ! the rows of a. A bit confusing, but correct, I think
  ! There are 4 cases to consider:
  ! 1. rowsa <= bufcols AND colsa <= bufrows
  ! 2. rowsa <= bufcols ( corresponds to a wide matrix )
  ! 3. colsa <= bufrows ( corresponds to a high matrix )
  ! 4. Both dimensions of a exceed both dimensions of the buffer
  ! 
  ! The main idea here is that the bufrows will define the usage of the
  ! L1 cache. We reference the same column or columns multiply while 
  ! accessing multiple partial rows of a transposed in the buffer.
  

  !
  !                 rowsa                           colsb
  !           <-bufca(1)>< (2) >                <-bufcb(1)><(2)>      
  !              i = 1, m  -ar->                   j = 1, n --br->
  !      ^    +----------+------+   ^          +----------+----+  ^
  !      |    |          x      |   |          |          x    |  |
  !      |    |          x      |   |          |          x    |  |
  !  bufr(1)  |  A**T    x      | rowchunks=2  |    B     x    |  |
  !      |    |          x      |   |          |          x    |  |
  !  |   |    | buffera  x      |   |    |     | bufferb  x    | ka = 1, k
  !  |   |    |          x      |   |    |     |          x    |  |
  ! ac   |    |    I     x III  |   |   bc     |    a     x c  |  |
  !  |   v    +xxxxxxxxxxxxxxxxx+   |    |     +xxxxxxxxxxxxxxx+  |
  !  v   ^    |          x      |   |    v     |          x    |  |
  !      |    |          x      |   |          |          x    |  |
  !   bufr(2) |          x      |   |          |          x    |  |
  !      |    |   II     x IV   |   |          |    b     x d  |  |
  !      V    +----------+------+   V          +----------+----+  V
  !            <--colachunks=2-->               <-colbchunks=2>
  !     x's mark buffer boudaries on the transposed matrices
  !     For this case, bufca(1) = bufcols, bufr(1) = bufrows

  colsa = kab
  rowsb = kab
  rowsa = mra
  colsb = ncb
  ! simple unbuffered multiplication for small matrices
  if (colsa * rowsa * colsb < min_blocked_mult) then
    if( beta .eq. 0.0 ) then
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0
            do k = 1, colsa
               temp0 = temp0 + alpha * a (i, k) * b( k, j )
            enddo
            c( i, j ) = temp0
         enddo
      enddo
    else
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0
            do k = 1, colsa
               temp0 = temp0 + alpha * a (i, k) * b( k, j )
            enddo
            c( i, j ) = beta * c( i, j ) + temp0
         enddo
      enddo
    endif
  else
    allocate( buffera( bufrows * bufcols ) )
    
    bufca = min( bufcols, rowsa ) 

    colachunks = ( rowsa + bufcols - 1)/bufcols
       ! set the number of buffer row chunks we will work on
    bufr = min( bufrows, colsa )
    bufr_sav = bufr
    rowchunks = ( colsa + bufr - 1 )/bufr 
    bufca_sav = bufca
    ac = 1   ! column index in matrix a for transpose
  !  lor = colsa - bufr ! left-over rows adjusts for the the fact that 
    ! colsa/bufr * bufr may not be equal to colsa 
    ! Note that the starting column index into matrix a (ac) is the same as 
    ! starting index into matrix b. But we need 1 less than that so we can
    ! add an index to it
    colsb_chunks = 4
    colsb_end = colsb/colsb_chunks * colsb_chunks
    colsb_strt = colsb_end + 1
    do rowchunk = 1, rowchunks
       ar = 1
       do colachunk = 1, colachunks
          bufca = min( bufca_sav, rowsa - ar + 1 )
          bufr = min( bufr_sav, colsa - ac + 1 )
          call ftn_transpose_real4( a( ar, ac ), lda, alpha, buffera, &
               & bufr, bufca )
          if ( ac .eq. 1 )then
             if( beta .eq. 0.0 ) then
                do j = 1, colsb_end, colsb_chunks
                   ndxa = 0
                   do i = ar, ar + bufca - 1
                      bk = ac - 1
                      temp0 = 0.0
                      temp1 = 0.0
                      temp2 = 0.0
                      temp3 = 0.0
                      do k = 1, bufr
                         bufatemp = buffera( ndxa + k )
                         temp0 = temp0 + bufatemp * b( bk + k, j )
                         temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                         temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                         temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                         ! temp4 = temp4 + bufatemp * b( bk + k, j + 4 )
                         ! temp5 = temp5 + bufatemp * b( bk + k, j + 5 )
                         ! temp6 = temp6 + bufatemp * b( bk + k, j + 6 )
                         ! temp7 = temp7 + bufatemp * b( bk + k, j + 7 )
                      enddo
                      c( i, j     ) = temp0
                      c( i, j + 1 ) = temp1
                      c( i, j + 2 ) = temp2
                      c( i, j + 3 ) = temp3
                      ndxa = ndxa + bufr
                   enddo
                enddo
                
                ! This takes care of the last colsb - colsb/colsb_chunks*colsb_chunks
                ! cases
                do j = colsb_strt, colsb
                   ndxa = 0
                   bk = ac - 1
                   do i = ar, ar + bufca - 1 
                      temp = 0.0
                      do k = 1, bufr
                         temp = temp + buffera( ndxa + k ) * b( bk + k, j )
                      enddo
                      c( i, j ) = temp
                      ndxa = ndxa + bufr
                   enddo
                enddo
             else
                do j = 1, colsb_end, colsb_chunks
                   ndxa = 0
                   do i = ar, ar + bufca - 1
                      bk = ac - 1
                      temp0 = 0.0
                      temp1 = 0.0
                      temp2 = 0.0
                      temp3 = 0.0
                      do k = 1, bufr
                         bufatemp = buffera( ndxa + k )
                         temp0 = temp0 + bufatemp * b( bk + k, j )
                         temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                         temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                         temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                         ! temp4 = temp4 + bufatemp * b( bk + k, j + 4 )
                         ! temp5 = temp5 + bufatemp * b( bk + k, j + 5 )
                         ! temp6 = temp6 + bufatemp * b( bk + k, j + 6 )
                         ! temp7 = temp7 + bufatemp * b( bk + k, j + 7 )
                      enddo
                      c( i, j     ) = beta * c( i, j )     + temp0
                      c( i, j + 1 ) = beta * c( i, j + 1 ) + temp1
                      c( i, j + 2 ) = beta * c( i, j + 2 ) + temp2
                      c( i, j + 3 ) = beta * c( i, j + 3 ) + temp3
                      ndxa = ndxa + bufr
                   enddo
                enddo
                
                ! This takes care of the last colsb - colsb/colsb_chunks*colsb_chunks
                ! cases
                do j = colsb_strt, colsb
                   ndxa = 0
                   bk = ac - 1
                   do i = ar, ar + bufca - 1 
                      temp = 0.0
                      do k = 1, bufr
                         temp = temp + buffera( ndxa + k ) * b( bk + k, j )
                      enddo
                      c( i, j ) = beta * c( i, j ) + temp
                      ndxa = ndxa + bufr
                   enddo
                enddo
             endif
          else
             do j = 1, colsb_end, colsb_chunks
                ndxa = 0
                do i = ar, ar + bufca - 1
                   bk = ac - 1
                   temp0 = 0.0
                   temp1 = 0.0
                   temp2 = 0.0
                   temp3 = 0.0
                   do k = 1, bufr
                      bufatemp = buffera( ndxa + k )
                      temp0 = temp0 + bufatemp * b( bk + k, j )
                      temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                      temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                      temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                      ! temp4 = temp4 + bufatemp * b( bk + k, j + 4 )
                      ! temp5 = temp5 + bufatemp * b( bk + k, j + 5 )
                      ! temp6 = temp6 + bufatemp * b( bk + k, j + 6 )
                      ! temp7 = temp7 + bufatemp * b( bk + k, j + 7 )
                   enddo
                   c( i, j     ) = c( i, j )     + temp0
                   c( i, j + 1 ) = c( i, j + 1 ) + temp1
                   c( i, j + 2 ) = c( i, j + 2 ) + temp2
                   c( i, j + 3 ) = c( i, j + 3 ) + temp3
                   ndxa = ndxa + bufr
                enddo
             enddo
             
             ! This takes care of the last colsb - colsb/colsb_chunks*colsb_chunks
             ! cases
             do j = colsb_strt, colsb
                ndxa = 0
                bk = ac - 1
                do i = ar, ar + bufca - 1 
                   temp = 0.0
                   do k = 1, bufr
                      temp = temp + buffera( ndxa + k ) * b( bk + k, j )
                   enddo
                   c( i, j ) = c( i, j ) + temp
                   ndxa = ndxa + bufr
                enddo
             enddo
          endif

          ! adjust the boundaries in the direction of the columns of a
          
          ar = ar + bufca
       enddo
       ! adjust the row values
       ac = ac + bufr
    enddo
    deallocate( buffera )
  endif
  return
end subroutine ftn_mnaxnb_real4
