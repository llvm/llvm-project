! @@name:	mem_model.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	rt-error
       PROGRAM EXAMPLE
       INCLUDE "omp_lib.h" ! or USE OMP_LIB
       INTEGER DATA
       INTEGER FLAG

       FLAG = 0
       !$OMP PARALLEL NUM_THREADS(2)
         IF(OMP_GET_THREAD_NUM() .EQ. 0) THEN
            ! Write to the data buffer that will be read by thread 1
            DATA = 42
            ! Flush DATA to thread 1 and strictly order the write to DATA
            ! relative to the write to the FLAG
            !$OMP FLUSH(FLAG, DATA)
            ! Set FLAG to release thread 1
            FLAG = 1;
            ! Flush FLAG to ensure that thread 1 sees the change */
            !$OMP FLUSH(FLAG)
         ELSE IF(OMP_GET_THREAD_NUM() .EQ. 1) THEN
            ! Loop until we see the update to the FLAG
            !$OMP FLUSH(FLAG, DATA)
            DO WHILE(FLAG .LT. 1)
               !$OMP FLUSH(FLAG, DATA)
            ENDDO

            ! Values of FLAG and DATA are undefined
            PRINT *, 'FLAG=', FLAG, ' DATA=', DATA
            !$OMP FLUSH(FLAG, DATA)

            !Values DATA will be 42, value of FLAG still undefined */
            PRINT *, 'FLAG=', FLAG, ' DATA=', DATA
         ENDIF
       !$OMP END PARALLEL
       END
