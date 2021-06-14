! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! Various checks with the nesting of SIMD construct

SUBROUTINE NESTED_GOOD(N)
  INTEGER N, I, J, K, A(10), B(10)
  !$OMP SIMD
  DO I = 1,N
    !$OMP ATOMIC
    K =  K + 1
    IF (I <= 10) THEN
      !$OMP ORDERED SIMD
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END SIMD

  !$OMP SIMD
  DO I = 1,N
    IF (I <= 10) THEN
      !$OMP SIMD
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END SIMD
    ENDIF
  END DO
  !$OMP END SIMD
END SUBROUTINE NESTED_GOOD

SUBROUTINE NESTED_BAD(N)
  INTEGER N, I, J, K, A(10), B(10)

  !$OMP SIMD
  DO I = 1,N
    IF (I <= 10) THEN
      !$OMP ORDERED SIMD
      DO J = 1,N
        print *, "Hi"
        !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
        !$omp teams 
         DO K = 1,N
        print *, 'Hello'
        END DO
        !$omp end teams
      END DO
      !$OMP END ORDERED
    ENDIF
  END DO
  !$OMP END SIMD

  !$OMP SIMD
  DO I = 1,N
    !$OMP ATOMIC
    K =  K + 1
    IF (I <= 10) THEN
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp task 
      do J = 1, N
        K = 2
      end do
      !$omp end task
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp teams 
      do J = 1, N
        K = 2
      end do
      !$omp end teams
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp target 
      do J = 1, N
        K = 2
      end do
      !$omp end target
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END DO
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP PARALLEL DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END PARALLEL DO
    ENDIF
  END DO
  !$OMP END SIMD

  !$OMP DO SIMD
  DO I = 1,N
    !$OMP ATOMIC
    K =  K + 1
    IF (I <= 10) THEN
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp task 
      do J = 1, N
        K = 2
      end do
      !$omp end task
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp teams 
      do J = 1, N
        K = 2
      end do
      !$omp end teams
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp target 
      do J = 1, N
        K = 2
      end do
      !$omp end target
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
      !$OMP DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END DO
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP PARALLEL DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END PARALLEL DO
    ENDIF
  END DO
  !$OMP END DO SIMD

  !$OMP PARALLEL DO SIMD
  DO I = 1,N
    !$OMP ATOMIC
    K =  K + 1
    IF (I <= 10) THEN
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp task 
      do J = 1, N
        K = 2
      end do
      !$omp end task
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp teams 
      do J = 1, N
        K = 2
      end do
      !$omp end teams
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp target 
      do J = 1, N
        K = 2
      end do
      !$omp end target
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
      !$OMP DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END DO
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP PARALLEL DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END PARALLEL DO
    ENDIF
  END DO
  !$OMP END PARALLEL DO SIMD

  !$OMP TARGET SIMD
  DO I = 1,N
    !$OMP ATOMIC
    K =  K + 1
    IF (I <= 10) THEN
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp task 
      do J = 1, N
        K = 2
      end do
      !$omp end task
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp teams 
      do J = 1, N
        K = 2
      end do
      !$omp end teams
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$omp target 
      do J = 1, N
        K = 2
      end do
      !$omp end target
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END DO
      !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
      !$OMP PARALLEL DO
      DO J = 1,N
        A(J) = J
      END DO
      !$OMP END PARALLEL DO
    ENDIF
  END DO
  !$OMP END TARGET SIMD


END SUBROUTINE NESTED_BAD
