! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

subroutine mod_task1(x)
  integer, intent(inout) :: x

  !Correct: "parallel" directive.
  !$omp parallel reduction(task, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end parallel
end

subroutine mod_task2(x)
  integer, intent(inout) :: x

  !Correct: worksharing directive.
  !$omp sections reduction(task, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end sections
end

subroutine mod_task3(x)
  integer, intent(inout) :: x

  !ERROR: Modifier 'TASK' on REDUCTION clause is only allowed with PARALLEL or worksharing directive
  !$omp simd reduction(task, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end simd
end

subroutine mod_inscan1(x)
  integer, intent(inout) :: x

  !Correct: worksharing-loop directive
  !$omp do reduction(inscan, +:x)
  do i = 1, 100
    !$omp scan inclusive(x)
    x = foo(i)
  enddo
  !$omp end do
end

subroutine mod_inscan2(x)
  integer, intent(inout) :: x

  !Correct: worksharing-loop simd directive
  !$omp do simd reduction(inscan, +:x)
  do i = 1, 100
    !$omp scan inclusive(x)
    x = foo(i)
  enddo
  !$omp end do simd
end

subroutine mod_inscan3(x)
  integer, intent(inout) :: x

  !Correct: "simd" directive
  !$omp simd reduction(inscan, +:x)
  do i = 1, 100
    !$omp scan inclusive(x)
    x = foo(i)
  enddo
  !$omp end simd
end

subroutine mod_inscan4(x)
  integer, intent(inout) :: x

  !ERROR: Modifier 'INSCAN' on REDUCTION clause is only allowed with WORKSHARING LOOP, WORKSHARING LOOP SIMD, or SIMD directive
  !$omp parallel reduction(inscan, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end parallel
end

subroutine mod_inscan5(x)
  integer, intent(inout) :: x

  !ERROR: Modifier 'INSCAN' on REDUCTION clause is only allowed with WORKSHARING LOOP, WORKSHARING LOOP SIMD, or SIMD directive
  !$omp sections reduction(inscan, +:x)
  do i = 1, 100
    x = foo(i)
  enddo
  !$omp end sections
end
