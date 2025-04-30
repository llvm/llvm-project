! @@name:	fort_loopvar.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
SUBROUTINE PLOOP_1(A,N)
INCLUDE "omp_lib.h"      ! or USE OMP_LIB

REAL A(*)
INTEGER I, MYOFFSET, N

!$OMP PARALLEL PRIVATE(MYOFFSET)
       MYOFFSET = OMP_GET_THREAD_NUM()*N
       DO I = 1, N
         A(MYOFFSET+I) = FLOAT(I)
       ENDDO
!$OMP END PARALLEL

END SUBROUTINE PLOOP_1
