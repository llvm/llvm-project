! @@name:	reduction.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
MODULE MOD
   INTRINSIC MAX, MIN
END MODULE MOD

PROGRAM REDUCTION4
   USE MOD, MIN=>MAX, MAX=>MIN
   REAL :: R
   R = -HUGE(0.0)

!$OMP PARALLEL DO REDUCTION(MIN: R)     ! still does MAX
   DO I = 1, 1000
      R = MIN(R, SIN(REAL(I)))
   END DO
   PRINT *, R
END PROGRAM REDUCTION4
