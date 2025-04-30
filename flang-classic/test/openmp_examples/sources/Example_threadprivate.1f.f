! @@name:	threadprivate.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      INTEGER FUNCTION INCREMENT_COUNTER()
        COMMON/INC_COMMON/COUNTER
!$OMP   THREADPRIVATE(/INC_COMMON/)

        COUNTER = COUNTER +1
        INCREMENT_COUNTER = COUNTER
        RETURN
      END FUNCTION INCREMENT_COUNTER
