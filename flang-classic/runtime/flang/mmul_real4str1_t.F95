! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 



#include "mmul_dir.h"

subroutine F90_matmul_real4_str1_t(dest,s1,s2, &
      k_extnt,m_extnt,n_extnt,                  &
      s1_d1_extnt,s2_d1_extnt,d_d1_extnt,       &
      d_d1_lstride)

  DESC_INT n_extnt,m_extnt,k_extnt
  DESC_INT s1_d1_extnt,s2_d1_extnt,d_d1_extnt,d_d1_lstride
  REAL*4 s1(s1_d1_extnt,m_extnt)
  REAL*4 s2(s2_d1_extnt,k_extnt)
  REAL*4 dest(d_d1_extnt,n_extnt*d_d1_lstride)

  DESC_INT k,n,m

  if (d_d1_lstride .eq. 1) then
     do k = 1, k_extnt
        do n = 1, n_extnt
           dest(n,k) = 0.0d0
        enddo
     enddo
     do k = 1, k_extnt
        do m = 1, m_extnt
           do n = 1, n_extnt
              dest(n,k) = dest(n,k) + s1(m,n) * s2(m,k)
           enddo
        enddo
     enddo
  else
     do k = 1, k_extnt
        do n = 1, n_extnt
           dest(1+(n-1)*d_d1_lstride,k) = 0.0d0
        enddo
     enddo
     do k = 1, k_extnt
       do m = 1, m_extnt
           do n = 1, n_extnt
              dest(1+(n-1)*d_d1_lstride,k) =                 &
                              dest(1+(n-1)*d_d1_lstride,k) + &
                                         s1(m,n) * s2(m,k)
           enddo
        enddo
     enddo
  endif
end subroutine


subroutine F90_matmul_real4_str1_mxv_t(dest, s1,s2,  &
                   n_extent,m_extent, ld1,dlstride) 

  DESC_INT n_extent,m_extent,ld1,ld2,dlstride
  REAL*4 s1(ld1,m_extent)
  REAL*4 s2(m_extent)
  REAL*4 dest(ld1)

  DESC_INT i,j,k

  if (dlstride .eq. 1) then
        do k = 1, m_extent
           dest(k) = 0.0d0
        enddo
        do j = 1, n_extent
           do k = 1, m_extent
              dest(k) = dest(k) + s1(j,k) * s2(j)
           enddo
        enddo
  else
        do k = 1, m_extent
           dest(1+(k-1)*dlstride) = 0.0d0
        enddo
        do j = 1, n_extent
           do k = 1, m_extent
              dest(1+(k-1)*dlstride) = dest(1+(k-1)*dlstride) +  &
                                       s1(j,k) * s2(j)
           enddo
        enddo
  endif
end
