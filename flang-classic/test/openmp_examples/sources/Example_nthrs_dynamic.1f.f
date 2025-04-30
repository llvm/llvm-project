! @@name:	nthrs_dynamic.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      PROGRAM EXAMPLE
        INCLUDE "omp_lib.h"      ! or USE OMP_LIB
        CALL OMP_SET_DYNAMIC(.FALSE.)
!$OMP     PARALLEL NUM_THREADS(10)
            ! do work here
!$OMP     END PARALLEL
      END PROGRAM EXAMPLE
