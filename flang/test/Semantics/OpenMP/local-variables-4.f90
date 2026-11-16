!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! A variable declared in a BLOCK inside an OpenMP construct has a predetermined
! data-sharing attribute for that construct.

subroutine s00
  !$omp parallel default(none)
  block
    integer :: cnt
    cnt = 0
    !$omp single
    cnt = cnt + 1
    !$omp end single
  end block
  !$omp end parallel
end

subroutine s01
  !$omp parallel default(none)
  block
    integer, save :: cnt
    !$omp single
    cnt = cnt + 1
    !$omp end single
  end block
  !$omp end parallel
end

subroutine s02
  !$omp parallel default(none)
  block
    integer :: v
    v = 0
    !$omp task firstprivate(v)
    v = v + 1
    !$omp end task
  end block
  !$omp end parallel
end

subroutine s03
  integer :: x
  !$omp parallel default(none)
  block
    !$omp single
    !ERROR: The DEFAULT(NONE) clause requires that 'x' must be listed in a data-sharing attribute clause
    x = 1
    !$omp end single
  end block
  !$omp end parallel
end

subroutine s04
  !$omp parallel default(none)
  block
    integer :: cnt
    !$omp task default(none)
    !ERROR: The DEFAULT(NONE) clause requires that 'cnt' must be listed in a data-sharing attribute clause
    cnt = 1
    !$omp end task
  end block
  !$omp end parallel
end

subroutine s05
  !$omp parallel
  block
    integer :: v
    !$omp single
    v = 7
    !$omp end single copyprivate(v)
    print *, v
  end block
  !$omp end parallel
end

subroutine s06
  !$omp parallel
  block
    integer, save :: s
    !$omp single
    s = 7
    !ERROR: COPYPRIVATE variable 's' is not PRIVATE or THREADPRIVATE in outer context
    !$omp end single copyprivate(s)
  end block
  !$omp end parallel
end
