! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! Check for cycle statements leaving an OpenMP structured block

program omp_do
  integer i, j, k

  !$omp parallel
  foo: do i = 0, 10
    !$omp do
    bar: do j = 0, 10
           !ERROR: CYCLE to construct 'foo' outside of DO construct is not allowed
           cycle foo
         end do bar
    !$omp end do
  end do foo
  !$omp end parallel

  foo1: do i = 0, 10
    !$omp parallel
    foo2: do k = 0, 10
      !$omp do
      foo3: do j = 0, 10
             !ERROR: CYCLE to construct 'foo1' outside of PARALLEL construct is not allowed
             !ERROR: CYCLE to construct 'foo1' outside of DO construct is not allowed
             cycle foo1
           end do foo3
      !$omp end do
      end do foo2
    !$omp end parallel
    end do foo1

  bar1: do i = 0, 10
    !$omp parallel
    bar2: do k = 0, 10
      bar3: do j = 0, 10
             !ERROR: CYCLE to construct 'bar1' outside of PARALLEL construct is not allowed
             cycle bar1
           end do bar3
      end do bar2
    !$omp end parallel
    end do bar1

end program omp_do
