!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 
!       PRIVATE variables
!       POINTER

      subroutine sub
      integer omp_get_thread_num
        integer, POINTER :: A( : )
        INTEGER err
        ALLOCATE( A( 5 : 6 ), STAT = err )
        IF ( err .NE. 0 ) THEN
          PRINT *, 'Error occured while allocating when serial'
          STOP
	ENDIF
	a(5) = 5
	a(6) = 6
!$OMP   PARALLEL PRIVATE( A, ii, err )
          ALLOCATE( A( 1 : 2 ), STAT = err )
          IF ( err .NE. 0 ) THEN
            PRINT *, 'Error occured while allocating when parallel'
            STOP
          END IF
          ii =  2*omp_get_thread_num()
          A(1) = ii+1
	  A(2) = ii+2
	  call stuff(a, ii+1)
          DEALLOCATE( A )
!$OMP   END PARALLEL
	call stuff(a,5)
        DEALLOCATE( A )
      END
      program test
      integer, dimension(6) :: expect = (/1,2,3,4,5,6/)
      integer result(6)
      common /result/result
      call omp_set_num_threads(2)
      call sub
      call check(result, expect, 6)
      end
      subroutine stuff(a,id)
      integer result(6)
      common /result/result
      integer a(2)
!$omp critical
      result(id) = a(1)
      result(id+1) = a(2)
!$omp endcritical
      end
