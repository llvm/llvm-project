! @@name:	tasking.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      MODULE LIST
         TYPE NODE
             INTEGER :: PAYLOAD
             TYPE (NODE), POINTER :: NEXT
         END TYPE NODE
      CONTAINS
          SUBROUTINE PROCESS(p)
             TYPE (NODE), POINTER :: P
                 ! do work here
          END SUBROUTINE
          SUBROUTINE INCREMENT_LIST_ITEMS (HEAD)
              TYPE (NODE), POINTER :: HEAD
              TYPE (NODE), POINTER :: P
              !$OMP PARALLEL PRIVATE(P)
                 !$OMP SINGLE
                      P => HEAD
                      DO
                         !$OMP TASK
                             ! P is firstprivate by default
                             CALL PROCESS(P)
                         !$OMP END TASK
                         P => P%NEXT
                         IF ( .NOT. ASSOCIATED (P) ) EXIT
                      END DO
                !$OMP END SINGLE
             !$OMP END PARALLEL
          END SUBROUTINE
       END MODULE
