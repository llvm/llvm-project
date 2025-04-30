! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"


subroutine ftn_mvmul_real4( ta, m, k, alpha, a, lda, b, beta, c )
  implicit none

  integer*8 :: m, k, lda
  real*4      :: alpha, beta
  real*4, dimension( lda, * ) :: a
  real*4, dimension( * )      :: b, c
  integer   :: ta
  ! Local variables
  
  integer*8   :: i, j, kk
  real*4      :: temp

  if( ta .eq. 0 )then ! normally oriented a matrix
     if( beta .ne. 0 )then
        do i = 1, m
           c( i ) = beta * c( i ) 
        enddo
     else
        do i = 1, m
           c( i ) = 0.0 
        enddo
     endif
     do kk = 1, k
        temp = alpha * b( kk )
        do i = 1, m
           c( i ) = c( i ) + a( i, kk ) * temp
        enddo
     enddo
  else ! matrix a is transposed - may be improved with buffering of b * alpha
     if( beta .ne. 0 )then
        do i = 1, m
           c( i ) = beta * c( i ) 
        enddo
     else
        do i = 1, m
           c( i ) = 0.0 
        enddo
     endif
     do i = 1, m
        temp = 0.0
        do kk = 1, k
           temp = temp + a( kk, i ) * b( kk )
        enddo
        c( i ) = c( i ) + alpha * temp
     enddo
  endif
  return
end subroutine ftn_mvmul_real4

