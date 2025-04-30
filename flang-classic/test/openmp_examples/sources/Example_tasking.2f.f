! @@name:	tasking.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
         RECURSIVE SUBROUTINE traverse ( P )
            TYPE Node
                TYPE(Node), POINTER :: left, right
             END TYPE Node
             TYPE(Node) :: P
             IF (associated(P%left)) THEN
                  !$OMP TASK   ! P is firstprivate by default
                      call traverse(P%left)
                  !$OMP END TASK
             ENDIF
             IF (associated(P%right)) THEN
                  !$OMP TASK    ! P is firstprivate by default
                      call traverse(P%right)
                  !$OMP END TASK
             ENDIF
             !$OMP TASKWAIT
             CALL process ( P )
          END SUBROUTINE
