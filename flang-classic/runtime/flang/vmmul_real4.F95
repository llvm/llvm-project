! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"


subroutine ftn_vmmul_real4( tb, n, k, alpha, a, b, ldb, beta, c )
  implicit none
  integer*8 :: n, k, ldb
  integer   :: tb
  real*4, dimension (ldb, * ) :: b
  real*4, dimension ( * )     :: a, c
  real*4                      :: alpha, beta

! local variables
  integer*8 :: i, j, kk
  real*4    :: temp        

  if( beta .ne. 0.0 )then
     do i = 1, n
        c( i ) = beta * c( i )
     enddo
  else
     do i = 1, n
        c( i ) = 0.0
     enddo
  end if
  if( tb .eq. 0 ) then !b is normally oriented
     if( alpha .eq. 1.0 )then
        do j = 1, n
           do kk = 1, k
              c( j ) = c( j ) + a( kk ) * b( kk, j )
           enddo
        enddo
     elseif( alpha .eq. -1.0 )then
        do j = 1, n
           do kk = 1, k
              c( j ) = c( j ) - a( kk ) * b( kk, j )
           enddo
        enddo
     else
        do j = 1, n
           do kk = 1, k
              c( j ) = c( j ) + alpha * a( kk ) * b( kk, j )
           enddo
        enddo
     endif
  else
     do kk = 1, k
        temp = alpha * a( kk )
        do j = 1, n
           c( j ) = c( j ) + temp * b( j, kk )
        enddo
     enddo
  endif
  return
end subroutine ftn_vmmul_real4
