! 
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
! 


#include "mmul_dir.h"

subroutine F90_matmul_cplx16_str1_mxv_t(dest, s1,s2, &
                   n_extent,m_extent, ld1,dlstride) 
   implicit none
   DESC_INT  n_extent,m_extent,ld1,ld2,dlstride
   COMPLEX*16 s1(ld1,m_extent)
   COMPLEX*16 s2(m_extent)
   COMPLEX*16 dest(ld1)
   DESC_INT  i,j,k
   DESC_INT  nmod8, mmod4, mmod2, n8, m2, kmod2, k2
   DESC_INT  jx,kx,ii,jj,kk,kmod4,k4,incx
   DESC_INT  j0,j1,j2,j3,ix,iy,jy,ky,m4
   COMPLEX*16 t0,t1,t2,t3,t4,t5,t6,t7,t8,x0,x1
   INTEGER   bs
   parameter (bs = 384)
   COMPLEX*16 temp (bs)
   DESC_INT   ind(bs)
   COMPLEX*16 zero
   parameter (zero = 0.0D0)

   if (dlstride .eq. 1) then
         do k = 1, m_extent
            dest(1+(k-1)*dlstride) = 0.0d0
         end do
         nmod8 = mod(m_extent, 8)
         mmod4 = mod(n_extent, 4)
         mmod2 = mod(n_extent, 2)
         n8 = m_extent - nmod8
         m4 = n_extent - mmod4
         m2 = n_extent- mmod2
         kx = 1
         ky = 1
         incx = 1
         jx = kx
         nmod8 = mod(m_extent, 8)
         n8= m_extent - nmod8
         do ii = 1, n_extent, bs
            jy = ky
            kk = 0
            ix = kx + (ii - 1) * incx
            do i = ii, min (n_extent, ii+bs-1)
               kk = kk + 1
               temp(kk) = s2(ix)
               ix = ix + incx
            end do
            kmod2 = mod(kk, 2)
            k2= kk - kmod2
         do j = 1, n8, 8
            t0 = zero
            t1 = zero
            t2 = zero
            t3 = zero
            t4 = zero
            t5 = zero
            t6 = zero
            t7 = zero
            do i = 1, k2, 2
               x0 = temp(i)
               x1 = temp(i + 1)
               t0 = t0 + s1(i+ii-1, j)*x0 &
                   + s1(i+ii, j)*x1
               t1 = t1 + s1(i+ii-1, j+1)*x0 &
                   + s1(i+ii, j+1)*x1
               t2 = t2 + s1(i+ii-1, j+2)*x0 &
                   + s1(i+ii, j+2)*x1
               t3 = t3 + s1(i+ii-1, j+3)*x0 &
                   + s1(i+ii, j+3)*x1
               t4 = t4 + s1(i+ii-1, j+4)*x0 &
                   + s1(i+ii, j+4)*x1
               t5 = t5 + s1(i+ii-1, j+5)*x0 &
                   + s1(i+ii, j+5)*x1
               t6 = t6 + s1(i+ii-1, j+6)*x0 &
                   + s1(i+ii, j+6)*x1
               t7 = t7 + s1(i+ii-1, j+7)*x0 &
                   + s1(i+ii, j+7)*x1
            end do
            do i = k2 + 1, kk 
               x0 = temp(i)
               t0 = t0 + s1(i+ii-1, j) * x0
               t1 = t1 + s1(i+ii-1, j+1)* x0
               t2 = t2 + s1(i+ii-1, j+2)* x0
               t3 = t3 + s1(i+ii-1, j+3)* x0
               t4 = t4 + s1(i+ii-1, j+4)* x0
               t5 = t5 + s1(i+ii-1, j+5)* x0
               t6 = t6 + s1(i+ii-1, j+6)* x0
               t7 = t7 + s1(i+ii-1, j+7)* x0
           end do
            dest(jy) = dest(jy) + t0
            dest(jy + 1) = dest(jy + 1) + t1
            dest(jy + 2) = dest(jy + 2) + t2
            dest(jy + 3) = dest(jy + 3) + t3
            dest(jy + 4) = dest(jy + 4) + t4
            dest(jy + 5) = dest(jy + 5) + t5
            dest(jy + 6) = dest(jy + 6) + t6
            dest(jy + 7) = dest(jy + 7) + t7
            jy = jy + 8 
         end do
         jy = ky + n8 
         do j = n8+1, m_extent
            t0 = zero
            do i = 1, k2, 2
               x0 = temp(i)
               x1 = temp(i + 1)
               t0 = t0 + s1(i+ii-1, j)*x0 &
                   + s1(i+ii, j)*x1
            end do
            dest(jy) = dest(jy) + t0
            jy= jy + 1
        end do

        jy = ky + n8 
        do j = n8+1, m_extent
           t0 = zero
           do i =  k2 + 1, kk
              x0 = temp(i)
              t0 = t0 + s1(i+ii-1, j)*x0
           end do
           dest(jy) = dest(jy) + t0
           jy= jy + 1
        end do
      end do

  else
         do k = 1, m_extent
            dest(1+(k-1)*dlstride) = 0.0d0
         end do
         nmod8 = mod(m_extent, 8)
         mmod4 = mod(n_extent, 4)
         mmod2 = mod(n_extent, 2)
         n8 = m_extent - nmod8
         m4 = n_extent - mmod4
         m2 = n_extent- mmod2
         kx = 1
         ky = 1
         incx = 1
         jx = kx
         nmod8 = mod(m_extent, 8)
         n8= m_extent - nmod8
         do ii = 1, n_extent, bs
            jy = ky
            kk = 0
            ix = kx + (ii - 1) * incx
            do i = ii, min (n_extent, ii+bs-1)
               kk = kk + 1
               temp(kk) = s2(ix)
               ix = ix + incx
            end do
            kmod2 = mod(kk, 2)
            k2= kk - kmod2
         do j = 1, n8, 8
            t0 = zero
            t1 = zero
            t2 = zero
            t3 = zero
            t4 = zero
            t5 = zero
            t6 = zero
            t7 = zero
            do i = 1, k2, 2
               x0 = temp(i)
               x1 = temp(i + 1)
               t0 = t0 + s1(i+ii-1, j)*x0 &
                   + s1(i+ii, j)*x1
               t1 = t1 + s1(i+ii-1, j+1)*x0 &
                   + s1(i+ii, j+1)*x1
               t2 = t2 + s1(i+ii-1, j+2)*x0 &
                   + s1(i+ii, j+2)*x1
               t3 = t3 + s1(i+ii-1, j+3)*x0 &
                   + s1(i+ii, j+3)*x1
               t4 = t4 + s1(i+ii-1, j+4)*x0 &
                   + s1(i+ii, j+4)*x1
               t5 = t5 + s1(i+ii-1, j+5)*x0 &
                   + s1(i+ii, j+5)*x1
               t6 = t6 + s1(i+ii-1, j+6)*x0 &
                   + s1(i+ii, j+6)*x1
               t7 = t7 + s1(i+ii-1, j+7)*x0 &
                   + s1(i+ii, j+7)*x1
            end do
            do i = k2 + 1, kk 
               x0 = temp(i)
               t0 = t0 + s1(i+ii-1, j) * x0
               t1 = t1 + s1(i+ii-1, j+1)* x0
               t2 = t2 + s1(i+ii-1, j+2)* x0
               t3 = t3 + s1(i+ii-1, j+3)* x0
               t4 = t4 + s1(i+ii-1, j+4)* x0
               t5 = t5 + s1(i+ii-1, j+5)* x0
               t6 = t6 + s1(i+ii-1, j+6)* x0
               t7 = t7 + s1(i+ii-1, j+7)* x0
           end do
            dest(1+(jy-1)*dlstride) = dest(1+(jy-1)*dlstride) + t0
            dest(1+(jy + 1 -1)*dlstride) = dest(1+(jy + 1 -1)*dlstride) + t1
            dest(1+(jy + 2 -1)*dlstride) = dest(1+(jy + 2 -1)*dlstride) + t2
            dest(1+(jy + 3 -1)*dlstride) = dest(1+(jy + 3 -1)*dlstride) + t3
            dest(1+(jy + 4 -1)*dlstride) = dest(1+(jy + 4 -1)*dlstride) + t4
            dest(1+(jy + 5 -1)*dlstride) = dest(1+(jy + 5 -1)*dlstride) + t5
            dest(1+(jy + 6 -1)*dlstride) = dest(1+(jy + 6 -1)*dlstride) + t6
            dest(1+(jy + 7 -1)*dlstride) = dest(1+(jy + 7 -1)*dlstride) + t7
            jy = jy + 8 
         end do
         jy = ky + n8 
         do j = n8+1, m_extent
            t0 = zero
            do i = 1, k2, 2
               x0 = temp(i)
               x1 = temp(i + 1)
               t0 = t0 + s1(i+ii-1, j)*x0 &
                   + s1(i+ii, j)*x1
            end do
            dest(1+(jy-1)*dlstride) = dest(1+(jy-1)*dlstride) + t0
            jy= jy + 1
        end do
   
        jy = ky + n8 
        do j = n8+1, m_extent
           t0 = zero
           do i =  k2 + 1, kk
              x0 = temp(i)
              t0 = t0 + s1(i+ii-1, j)*x0
           end do
           dest(1+(jy-1)*dlstride) = dest(1+(jy-1)*dlstride) + t0
           jy= jy + 1
        end do
      end do
  endif
end



