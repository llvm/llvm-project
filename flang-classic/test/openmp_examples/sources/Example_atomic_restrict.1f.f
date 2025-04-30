! @@name:	atomic_restrict.1f
! @@type:	F-fixed
! @@compilable:	maybe
! @@linkable:	no
! @@expect:	failure
      SUBROUTINE ATOMIC_WRONG()
        INTEGER:: I
        REAL:: R
        EQUIVALENCE(I,R)

!$OMP   PARALLEL
!$OMP     ATOMIC UPDATE
            I = I + 1
!$OMP     ATOMIC UPDATE
            R = R + 1.0
! incorrect because I and R reference the same location
! but have different types
!$OMP   END PARALLEL
      END SUBROUTINE ATOMIC_WRONG
