! @@name:	mem_model.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	rt-error
       PROGRAM EXAMPLE
       INCLUDE "omp_lib.h" ! or USE OMP_LIB
       INTEGER FLAG

       FLAG = 0
       !$OMP PARALLEL NUM_THREADS(3)
         IF(OMP_GET_THREAD_NUM() .EQ. 0) THEN
             ! Set flag to release thread 1
             !$OMP ATOMIC UPDATE
                 FLAG = FLAG + 1
             !Flush of FLAG is implied by the atomic directive
         ELSE IF(OMP_GET_THREAD_NUM() .EQ. 1) THEN
             ! Loop until we see that FLAG reaches 1
             !$OMP FLUSH(FLAG, DATA)
             DO WHILE(FLAG .LT. 1)
                 !$OMP FLUSH(FLAG, DATA)
             ENDDO

             PRINT *, 'Thread 1 awoken'

             ! Set FLAG to release thread 2
             !$OMP ATOMIC UPDATE
                 FLAG = FLAG + 1
             !Flush of FLAG is implied by the atomic directive
         ELSE IF(OMP_GET_THREAD_NUM() .EQ. 2) THEN
             ! Loop until we see that FLAG reaches 2
             !$OMP FLUSH(FLAG, DATA)
             DO WHILE(FLAG .LT. 2)
                 !$OMP FLUSH(FLAG,    DATA)
             ENDDO

             PRINT *, 'Thread 2 awoken'
         ENDIF
       !$OMP END PARALLEL
       END
