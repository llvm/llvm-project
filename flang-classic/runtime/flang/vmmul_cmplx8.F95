! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


! directives.h -- contains preprocessor directives for F90 rte files
#include "mmul_dir.h"

subroutine ftn_vmmul_cmplx8( ta, tb, n, k, alpha, a, b, ldb, beta, c )
  implicit none
  integer*8 :: n, k, ldb
  integer   :: ta, tb
  complex*8, dimension (ldb, * ) :: b
  complex*8, dimension ( * )     :: a, c
  complex*8                      :: alpha, beta

! local variables
  integer*8 :: i, j, kk
  complex*8    :: temp        

   
!  print *, "#### In vmmul ####"

  if( beta .ne. 0.0 )then
     do i = 1, n
        c( i ) = beta * c( i )
     enddo
  else
     do i = 1, n
        c( i ) = 0.0
     enddo
  end if
 



  
  if( tb .eq. 2 )then !conjugate b
     if( ta .eq. 2 )then ! conjugate a - since tb = 2, b is normally oriented
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + conjg( a( kk ) ) * conjg( b( j, kk ) )
              enddo
           enddo
        elseif( alpha .eq. (-1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) - conjg( a( kk ) ) * conjg( b( j, kk ) )
              enddo
           enddo
        else
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + alpha * conjg( a( kk ) ) * conjg( b( kk, j ) )
              enddo
           enddo
        endif
     else ! don't conjugate a - if ta != 2, it is just a complex vector
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + a( kk ) * conjg( b( j, kk ) )
              enddo
           enddo
        elseif( alpha .eq. ( -1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) - a( kk ) * conjg( b( j, kk ) )
              enddo
           enddo
        else
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + alpha * a( kk ) * conjg( b( j, kk ) )
              enddo
           enddo
        endif
     endif
  elseif( tb .eq. 1 )then ! b is tranpsosed
     if( ta .ne. 2 )then ! no conjugation of a is required
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) +  a( kk ) * b( j, kk )
              enddo
           enddo
        elseif( alpha .eq. ( -1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) - a( kk ) * b( j, kk )
              enddo
           enddo
        else
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + alpha * a( kk ) * b( j, kk )
              enddo
           enddo
        endif
     else
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) +  conjg( a( kk ) ) * b( j, kk )
              enddo
           enddo
        elseif( alpha .eq. ( -1.0, 0.0 ) )then
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) - conjg( a( kk ) ) * b( j, kk )
              enddo
           enddo
        else
           do j = 1, n
              do kk = 1, k
                 c( j ) = c( j ) + alpha * conjg( a( kk ) ) * b( j, kk )
              enddo
           enddo
        endif
     endif
  else ! b is normally oriented - 
     if( ta .ne. 2 )then ! a is not conjugated
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do j = 1, n
              temp = 0.0
              do kk = 1, k
                 temp = temp + a(kk) * b( kk, j )
              enddo
              c( j ) = c( j ) + temp
           enddo
        elseif( alpha .eq. ( -1.0, 0.0 ) )then
           do j = 1, n
              temp = 0.0
              do kk = 1, k
                 temp = temp + a(kk) * b( kk, j )
              enddo
              c( j ) = c( j ) - temp
           enddo
        else
           do j = 1, n
              temp = 0.0
              do kk = 1, k
                 temp = temp + a(kk) * b( kk, j )
              enddo
              c( j ) = c( j ) + alpha * temp
           enddo
        endif
     else ! a is conjugated
        if( alpha .eq. ( 1.0, 0.0 ) )then
           do kk = 1, k
              temp = conjg( a( kk ) )
              do j = 1, n
                 c( j ) = c( j ) + temp * b( j, kk )
              enddo
           enddo
        elseif( alpha .eq. ( -1.0, 0.0 ) )then
           do kk = 1, k
              temp = conjg( a( kk ) )
              do j = 1, n
                 c( j ) = c( j ) - temp * b( j, kk )
              enddo
           enddo
        else
           do kk = 1, k
              temp = alpha * conjg( a( kk ) )
              do j = 1, n
                 c( j ) = c( j ) - temp * b( j, kk )
              enddo
           enddo
        endif
     endif
  endif
return
end subroutine ftn_vmmul_cmplx8
