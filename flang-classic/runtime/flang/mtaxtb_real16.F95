!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

#include "mmul_dir.h"


subroutine ftn_mtaxtb_real16( mra, ncb, kab, alpha, a, lda, b, ldb, beta, &
     & c, ldc )
  implicit none
#include "pgf90_mmul_real16.h"

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
  !                 rowsa                           <-bufcb->
  !                                                   colsb
  !              i = 1, m  -ac->                   j = 1, n --bc->
  !      |    +-----------------+   ^          +----------+----+  ^
  !      |    |                 |   |          |          x    |  |
  !      |    |                 |   |          |          x    |  |
  !    ak     |  A              | rowchunks=2  |    B     x    |  |
  !      |    |                 |   |          |          x    |  |
  !      |    |                 |   |    |     |          x    | ka = 1, k
  !      |    |                 |   |    |     |          x    |  |
  !      |    |                 |   |   br     |    a     x    |  |
  !      v    +xxxxxxxxxxxxxxxxx+   |    |     +xxxxxxxxxxxxxxx+  |
  !      |    |                 |   |    v     |          x    |  |
  !      |    |                 |   |          |          x    |  |
  !           |                 |   |          |          x    |  |
  !      |    |                 |   |          |    b     x    |  |
  !      V    +-----------------+   V          +----------+----+  V
  !            <--colachunks=2-->
  !     x's mark buffer boudaries on the transposed matrices
  !     For this case, bufca(1) = bufcols, bufr(1) = bufrows

  colsa = kab
  rowsb = kab
  rowsa = mra
  colsb = ncb
  if (colsa * rowsa * colsb < min_blocked_mult) then
    if( beta .eq. 0.0_16 ) then
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0_16
            do k = 1, colsa
               temp0 = temp0 + alpha * a (k, i) * b( j, k )
            enddo
            c( i, j ) = temp0
         enddo
      enddo
    else
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0_16
            do k = 1, colsa
               temp0 = temp0 + alpha * a (k, i) * b( j, k )
            enddo
            c( i, j ) = beta * c( i, j ) + temp0
         enddo
      enddo
    endif
  else
    allocate( bufferb( bufrows * bufcols ) )

       ! set the number of buffer row chunks we will work on
    bufr = min( bufrows, rowsb )
    bufr_sav = bufr
    rowchunks = ( rowsb + bufr - 1 )/bufr

    bufcb = min( bufcols, colsb )
    bufcb_sav = bufcb
    colbchunks = ( colsb + bufcb - 1)/bufcb
    ! Note that the starting column index into matrix a (ac) is the same as
    ! starting index into matrix b. But we need 1 less than that so we can
    ! add an index to it
    br = 1
    ac = 1
    bc = 1
    ak = 0
    colsa_chunk = 4
    colsa_chunks = mra/colsa_chunk
    colsa_end = colsa_chunks * colsa_chunk
    colsa_strt = colsa_end + 1


    do rowchunk = 1, rowchunks
        bc = 1
        do colbchunk = 1, colbchunks
           ak = br - 1
           if( br .eq. 1 ) then
             bufcb = min( bufcb_sav, colsb - bc + 1 )
             bufr = min( bufr_sav, rowsb - br + 1 )
             call ftn_transpose_real16( b( bc, br ), ldb, alpha, bufferb, &
                  & bufr, bufcb )
             if( beta .eq. 0.0_16 ) then
                do i = 1, colsa_end, colsa_chunk
                   ndxb = 0
                   do j = bc, bc + bufcb - 1
                      temp0 = 0.0_16
                      temp1 = 0.0_16
                      temp2 = 0.0_16
                      temp3 = 0.0_16
                      do k = 1, bufr
                         bufbtemp = bufferb( ndxb + k )
                         temp0 = temp0 + bufbtemp * a( ak + k, i )
                         temp1 = temp1 + bufbtemp * a( ak + k, i + 1 )
                         temp2 = temp2 + bufbtemp * a( ak + k, i + 2 )
                         temp3 = temp3 + bufbtemp * a( ak + k, i + 3 )
                      enddo
                      c( i, j )     = temp0
                      c( i + 1, j ) = temp1
                      c( i + 2, j ) = temp2
                      c( i + 3, j ) = temp3
                      ndxb = ndxb + bufr
                   enddo
                enddo
                ! Now clean up whatever is left from the loop unrolling
                do i = colsa_strt, mra
                   ndxb = 0
                   do j = bc, bc + bufcb - 1
                      temp = 0.0_16
                      do k = 1, bufr
                         temp = temp + bufferb( ndxb + k ) * a( ak + k, i )
                      enddo
                      c( i, j ) = temp
                      ndxb = ndxb + bufr
                   enddo
                enddo
             else
                do i = 1, colsa_end, colsa_chunk
                   ndxb = 0
                   do j = bc, bc + bufcb - 1
                      temp0 = 0.0_16
                      temp1 = 0.0_16
                      temp2 = 0.0_16
                      temp3 = 0.0_16
                      do k = 1, bufr
                         bufbtemp = bufferb( ndxb + k )
                         temp0 = temp0 + bufbtemp * a( ak + k, i )
                         temp1 = temp1 + bufbtemp * a( ak + k, i + 1 )
                         temp2 = temp2 + bufbtemp * a( ak + k, i + 2 )
                         temp3 = temp3 + bufbtemp * a( ak + k, i + 3 )
                      enddo
                      c( i, j )     = beta * c( i, j )     + temp0
                      c( i + 1, j ) = beta * c( i + 1, j ) + temp1
                      c( i + 2, j ) = beta * c( i + 2, j ) + temp2
                      c( i + 3, j ) = beta * c( i + 3, j ) + temp3
                      ndxb = ndxb + bufr
                   enddo
                enddo
                ! Now clean up whatever is left from the loop unrolling
                do i = colsa_strt, mra
                   ndxb = 0
                   do j = bc, bc + bufcb - 1
                      temp = 0.0_16
                      do k = 1, bufr
                         temp = temp + bufferb( ndxb + k ) * a( ak + k, i )
                      enddo
                      c( i, j ) = beta * c( i, j ) + temp
                      ndxb = ndxb + bufr
                   enddo
                enddo
             endif
          else
             bufcb = min( bufcb_sav, colsb - bc + 1 )
             bufr = min( bufr_sav, rowsb - br + 1 )
             call ftn_transpose_real16( b( bc, br ), ldb, alpha, bufferb, &
                  & bufr, bufcb )
             do i = 1, colsa_end, colsa_chunk
                ndxb = 0
                do j = bc, bc + bufcb - 1
                   temp0 = 0.0_16
                   temp1 = 0.0_16
                   temp2 = 0.0_16
                   temp3 = 0.0_16
                   do k = 1, bufr
                      bufbtemp = bufferb( ndxb + k )
                      temp0 = temp0 + bufbtemp * a( ak + k, i )
                      temp1 = temp1 + bufbtemp * a( ak + k, i + 1 )
                      temp2 = temp2 + bufbtemp * a( ak + k, i + 2 )
                      temp3 = temp3 + bufbtemp * a( ak + k, i + 3 )
                   enddo
                   c( i, j )     = c( i, j )     + temp0
                   c( i + 1, j ) = c( i + 1, j ) + temp1
                   c( i + 2, j ) = c( i + 2, j ) + temp2
                   c( i + 3, j ) = c( i + 3, j ) + temp3
                   ndxb = ndxb + bufr
                enddo
             enddo
             ! Now clean up whatever is left from the loop unrolling
             do i = colsa_strt, mra
                ndxb = 0
                do j = bc, bc + bufcb - 1
                   temp = 0.0_16
                   do k = 1, bufr
                      temp = temp + bufferb( ndxb + k ) * a( ak + k, i )
                   enddo
                   c( i, j ) = c( i, j ) + temp
                   ndxb = ndxb + bufr
                enddo
             enddo
          endif
          ! adjust the boundaries in the direction of the columns of b
          ! adjust the row values
          bc = bc + bufcb
       enddo
       br = br + bufr
       ! controlled but tcbe numbebrcbof bufferb chunks we use.

    enddo
    deallocate( bufferb )
  endif
  return
end subroutine ftn_mtaxtb_real16
