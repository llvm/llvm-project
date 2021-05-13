! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! Various checks with the ordered construct

SUBROUTINE WORK(I)
  INTEGER I
END SUBROUTINE WORK

SUBROUTINE ORDERED_GOOD(N)
  INTEGER N, I, A(10), B(10), C(10) 
  !$OMP SIMD
  DO I = 1,N
    IF (I <= 10) THEN
      !$OMP ORDERED SIMD
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END SIMD
END SUBROUTINE ORDERED_GOOD

SUBROUTINE ORDERED_BAD(N)
  INTEGER N, I, A(10), B(10), C(10)

  !$OMP DO SIMD
  DO I = 1,N
    IF (I <= 10) THEN
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !ERROR: The ORDERED clause must be present on the loop construct if any ORDERED region ever binds to a loop region arising from the loop construct.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END DO SIMD

  !$OMP PARALLEL DO
  DO I = 1,N
    IF (I <= 10) THEN
      !ERROR: The ORDERED clause must be present on the loop construct if any ORDERED region ever binds to a loop region arising from the loop construct.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END PARALLEL DO

  !$OMP CRITICAL  
  DO I = 1,N
    IF (I <= 10) THEN
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END CRITICAL

  !$OMP CRITICAL
    WRITE(*,*) I
    !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
    !$OMP ORDERED 
    CALL WORK(I)
    !$OMP END ORDERED
  !$OMP END CRITICAL

  !$OMP ORDERED 
    WRITE(*,*) I
    IF (I <= 10) THEN
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  !$OMP END ORDERED

  !$OMP TASK  
    C =  C - A * B
    !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
    !$OMP ORDERED 
    CALL WORK(I)
    !$OMP END ORDERED
  !$OMP END TASK

  !$OMP TASKLOOP 
  DO I = 1,N
    IF (I <= 10) THEN
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END TASKLOOP

  !$OMP CRITICAL  
    C =  C - A * B
    !$OMP MASTER
    DO I = 1,N
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    END DO
    !$OMP END MASTER
  !$OMP END CRITICAL

  !$OMP ORDERED  
    C =  C - A * B
    !$OMP MASTER
    DO I = 1,N
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    END DO
    !$OMP END MASTER
  !$OMP END ORDERED

  !$OMP TASK  
    C =  C - A * B
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$OMP MASTER
    DO I = 1,N
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    END DO
    !$OMP END MASTER
  !$OMP END TASK

  !$OMP TASKLOOP
  DO J= 1,N  
    C =  C - A * B
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$OMP MASTER
    DO I = 1,N
      !ERROR: `ORDERED` region may not be closely nested inside of `CRITICAL`, `ORDERED`, explicit `TASK` or `TASKLOOP` region.
      !$OMP ORDERED 
      CALL WORK(I)
      !$OMP END ORDERED
    END DO
    !$OMP END MASTER
  END DO
  !$OMP END TASKLOOP

END SUBROUTINE ORDERED_BAD
