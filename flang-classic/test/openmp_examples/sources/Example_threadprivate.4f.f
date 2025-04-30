! @@name:	threadprivate.4f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
       SUBROUTINE INC_GOOD()
        COMMON /T/ A
!$OMP   THREADPRIVATE(/T/)

        CONTAINS
          SUBROUTINE INC_GOOD_SUB()
            COMMON /T/ A
!$OMP       THREADPRIVATE(/T/)

!$OMP       PARALLEL COPYIN(/T/)
!$OMP       END PARALLEL
         END SUBROUTINE INC_GOOD_SUB
       END SUBROUTINE INC_GOOD
