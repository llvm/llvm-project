! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"


subroutine ftn_mtaxnb_real8( mra, ncb, kab, alpha, a, lda, b, ldb, beta, &
     & c, ldc )
  implicit none
#include "pgf90_mmul_real8.h"
  real*8, dimension( mra, kab )::at
  

  !
  !                 rowsa                          
  !           <-bufca(1)>< (2) >                       colsb  
  !              i = 1, m  -ar->                   j = 1, n 
  !      ^    +----------+------+   ^  bk = 0->+--------------------+  ^
  !      |    |          x      |   |          |                    |  |
  !      |    |          x      |   |          |                    |  |
  !  bufr(1)  |  A**T    x      | rowchunks=2  |                    |  |
  !      |    |          x      |   |          |         B          |  |
  !  |   |    | buffera  x      |   |          |                    | ka = 1, k
  !  |   |    |          x      |   |          |                    |  |    
  ! ac   |    |    I     x III  |   |          |                    |  |
  !  |   v    +xxxxxxxxxxxxxxxxx+   |  bk = bk>+xxxxxxxxxxxxxxxxxxxx+  |
  !  v   ^    |          x      |   |   + bufr |                    |  |
  !      |    |          x      |   |          |                    |  |
  !   bufr(2) |          x      |   |          |                    |  |
  !      |    |   II     x IV   |   |          |                    |  |
  !      V    +----------+------+   V          +--------------------+  V
  !            <--colachunks=2-->              
  !     x's mark buffer boudaries on the transposed matrix for A, the
  !     part of B that is multiplied by buffera in B
  !     


  !( I think this comment should be removed. The exchange of meanings for
  ! colsa and rowsa is valid IF you are simply writing DO loops, but
  ! we are not doing that herein.
  ! since matrix a is transposed, the rows and columns get switched
  colsa = kab
  rowsb = kab
  rowsa = mra
  colsb = ncb
    !call ftn_transpose_real8( a, lda, alpha, at, colsb, colsa) 
  if (colsa * rowsa * colsb < min_blocked_mult) then
    if( beta .eq. 0.0 ) then
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0
            do k = 1, colsa
               temp0 = temp0 + alpha * a (k, i) * b( k, j )
            enddo
            c( i, j ) = temp0
         enddo
      enddo
    else
      do j = 1, colsb
         do i = 1, rowsa
            temp0 = 0.0
            do k = 1, colsa
               temp0 = temp0 + alpha * a (k, i) * b( k, j )
            enddo
            c( i, j ) = beta * c( i, j ) + temp0
         enddo
      enddo
    endif
  else
    allocate( buffera( bufrows * bufcols ) )

    bufca = min( rowsa, bufcols )
    bufca_sav = bufca
    colachunks = ( rowsa + bufca - 1)/bufca
    ! set the number of buffer row chunks we will work on
    bufr = min( colsa, bufrows )
    bufr_sav = bufr
    rowchunks = ( colsa + bufr - 1 )/bufr 
       
    ac = 1   ! column index in matrix a for gather. 
    ! Note that the starting column index into matrix a (ac) is the same as 
    ! starting index into matrix b. But we need 1 less than that so we can
    ! add an index to it
    ar = 1
    colsb_chunk = 4
    colsb_chunks = colsb/colsb_chunk
    colsb_end = colsb_chunks * colsb_chunk
    colsb_strt = colsb_end + 1

    do rowchunk = 1, rowchunks ! This will set the values over k
       ar = 1 ! row index in matrix a for gather and reference to C()
       !     loc = rowsa - bufca
       do colachunk = 1, colachunks ! this over m     
          if( ac .eq. 1 ) then
             bufca = min( bufca_sav, rowsa - ar + 1 )
             bufr = min( bufr_sav, colsa - ac + 1 )
             call ftn_gather_real8( a( ac, ar ), lda, alpha,  buffera, &
                  & bufr, bufca )
             bk = ac - 1
             if( beta .eq. 0.0 ) then
                do j = 1, colsb_end, colsb_chunk
                   ndxa = 0
                   do i = ar, ar + bufca - 1 
                      temp0 = 0
                      temp1 = 0
                      temp2 = 0
                      temp3 = 0
                      do k = 1, bufr
                         bufatemp = buffera( ndxa + k )
                         temp0 = temp0 + bufatemp * b( bk + k, j )
                         temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                         temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                         temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                      enddo
                      c( i, j )     = temp0
                      c( i, j + 1 ) = temp1
                      c( i, j + 2 ) = temp2
                      c( i, j + 3 ) = temp3
                      ndxa = ndxa + bufr
                   enddo
                enddo
                do j = colsb_strt, colsb
                   ndxa = 0
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
                do j = 1, colsb_end, colsb_chunk
                   ndxa = 0
                   do i = ar, ar + bufca - 1 
                      temp0 = 0
                      temp1 = 0
                      temp2 = 0
                      temp3 = 0
                      do k = 1, bufr
                         bufatemp = buffera( ndxa + k )
                         temp0 = temp0 + bufatemp * b( bk + k, j )
                         temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                         temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                         temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                      enddo
                      c( i, j )     = beta * c( i, j )     + temp0
                      c( i, j + 1 ) = beta * c( i, j + 1 ) + temp1
                      c( i, j + 2 ) = beta * c( i, j + 2 ) + temp2
                      c( i, j + 3 ) = beta * c( i, j + 3 ) + temp3
                      ndxa = ndxa + bufr
                   enddo
                enddo
                do j = colsb_strt, colsb
                   ndxa = 0
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
             bufca = min( bufca_sav, rowsa - ar + 1 )
             bufr = min( bufr_sav, colsa - ac + 1 )
             call ftn_gather_real8( a( ac, ar ), lda, alpha,  buffera, &
                  & bufr, bufca )
             bk = ac - 1
             do j = 1, colsb_end, colsb_chunk
                ndxa = 0
                do i = ar, ar + bufca - 1 
                   temp0 = 0
                   temp1 = 0
                   temp2 = 0
                   temp3 = 0
                   do k = 1, bufr
                      bufatemp = buffera( ndxa + k )
                      temp0 = temp0 + bufatemp * b( bk + k, j )
                      temp1 = temp1 + bufatemp * b( bk + k, j + 1 )
                      temp2 = temp2 + bufatemp * b( bk + k, j + 2 )
                      temp3 = temp3 + bufatemp * b( bk + k, j + 3 )
                   enddo
                   c( i, j )     = c( i, j )     + temp0
                   c( i, j + 1 ) = c( i, j + 1 ) + temp1
                   c( i, j + 2 ) = c( i, j + 2 ) + temp2
                   c( i, j + 3 ) = c( i, j + 3 ) + temp3
                   ndxa = ndxa + bufr
                enddo
             enddo
             do j = colsb_strt, colsb
                ndxa = 0
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
          ar = ar + bufca
  !        bufr = min( bufr, lor )
  !        lor = lor - bufr 
       enddo
       ac = ac + bufr
  !     bufca = min( bufca, loc )
  !     loc = loc - bufca ! Note: this is not circular since the loops are 
       ! controlled but the number of buffera chunks we use.
  !     bufr = bufr + colsa

  !     lor = colsa - bufr
    enddo
    
    deallocate( buffera )
  endif
  return
end subroutine ftn_mtaxnb_real8
