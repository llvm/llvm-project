! @@name:	copyprivate.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
        REAL FUNCTION READ_NEXT()
        REAL, POINTER :: TMP

!$OMP   SINGLE
          ALLOCATE (TMP)
!$OMP   END SINGLE COPYPRIVATE (TMP)  ! copies the pointer only

!$OMP   MASTER
          READ (11) TMP
!$OMP   END MASTER

!$OMP   BARRIER
          READ_NEXT = TMP
!$OMP   BARRIER

!$OMP   SINGLE
          DEALLOCATE (TMP)
!$OMP   END SINGLE NOWAIT
        END FUNCTION READ_NEXT
