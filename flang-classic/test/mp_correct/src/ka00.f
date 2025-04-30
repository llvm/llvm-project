!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Optimizer bug 
*   _mp_lcpu2() & _mp_ncpus() cannot be invariant (must call this routines
*   using JSR and not QJSR.

      program mmul
c
      integer i, j, k
      integer m, n, p
      real*8 a, b, c, arow
      allocatable a(:,:), b(:,:), c(:,:), arow(:)
c
      integer ntimes, l
      real time, mflops
      integer hz, clock0, clock1, clock2
      integer t
      allocatable t(:)
      real tarray(2), time0, time1, time2
      allocate (t(1))
c
      m = 100
      n = 100
      p = 100
      ntimes = 10
c
      allocate (a(1:m,1:n),b(1:n,1:p),c(1:m,1:p),arow(1:n))
c
      do i = 1, m
      do j = 1, n
         a(i,j) = 1.0
      enddo
      enddo
      do i = 1, n
      do j = 1, p
         b(i,j) = 1.0
      enddo
      enddo
c
      call mmul_time(a, b, c, m, n, p, ntimes, arow,
     &               clock0, clock1, clock2, hz,
     &               time0, time1, time2, tarray)
c
!      print *, nint(c(1,1))	!should be ~100.0
      call check(nint(c(1,1)), 100, 1)
c
      end

      subroutine mmul_time (a, b, c, m, n, p, ntimes, arow,
     &                      clock0, clock1, clock2, hz,
     &                      time0, time1, time2, tarray)
      integer m, n, p
      real*8 a(m,n), b(n,p), c(m,p), arow(n)
      integer clock0, clock1, clock2, hz
      real tarray(2), time0, time1, time2
      real ettime
      time0 = ettime(tarray)
      do l = 1, ntimes
         do i = 1, m
            do j = 1, p
               c(i,j) = 0.0
            enddo
         enddo
!$omp parallel
!$omp do
         do i = 1, m
            do j = 1, p
               do k = 1, n
                  c(i,j) = c(i,j) + a(i,k) * b(k,j)
               enddo
            enddo
         enddo
!$omp end parallel
         call dummy(c)
      enddo
      time1 = ettime(tarray)
      do i = 1, ntimes
         call dummy(c)
      enddo
      time2 = ettime(tarray)
      return
      end
c
c Dummy subroutine
c
      subroutine dummy(c)
      return
      end
      real function ettime(tarray)
      real tarray(2)
      ettime = 0
      end
