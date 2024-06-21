! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 2.18.2 Cancellation Point Construct

program main
  integer :: i, N = 10
  real :: a

  !ERROR: CANCELLATION POINT TASKGROUP directive is not closely nested inside TASK or TASKLOOP
  !$omp cancellation point taskgroup

  !ERROR: CANCELLATION POINT SECTIONS directive is not closely nested inside SECTION or SECTIONS
  !$omp cancellation point sections

  !ERROR: CANCELLATION POINT DO directive is not closely nested inside the construct that matches the DO clause type
  !$omp cancellation point do

  !ERROR: CANCELLATION POINT PARALLEL directive is not closely nested inside the construct that matches the PARALLEL clause type
  !$omp cancellation point parallel

  !$omp parallel
  !$omp sections
  !$omp cancellation point sections
  !$omp section
  a = 3.14
  !$omp end sections
  !$omp end parallel

  !$omp sections
  !$omp section
  !$omp cancellation point sections
  a = 3.14
  !$omp end sections

  !$omp parallel
  !ERROR: With SECTIONS clause, CANCELLATION POINT construct cannot be closely nested inside PARALLEL construct
  !$omp cancellation point sections
  a = 3.14
  !$omp end parallel

  !$omp parallel sections
  !$omp cancellation point sections
  a = 3.14
  !$omp end parallel sections

  !$omp do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end do

  !$omp parallel do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end parallel do

  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end distribute parallel do
  !$omp end teams
  !$omp end target

  !$omp target
  !$omp teams distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end teams distribute parallel do
  !$omp end target

  !$omp target teams distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end target teams distribute parallel do

  !$omp target parallel do
  do i = 1, N
    a = 3.14
    !$omp cancellation point do
  end do
  !$omp end target parallel do

  !$omp parallel
  do i = 1, N
    a = 3.14
    !ERROR: With DO clause, CANCELLATION POINT construct cannot be closely nested inside PARALLEL construct
    !$omp cancellation point do
  end do
  !$omp end parallel

  !$omp parallel
  do i = 1, N
    a = 3.14
    !$omp cancellation point parallel
  end do
  !$omp end parallel

  !$omp target parallel
  do i = 1, N
    a = 3.14
    !$omp cancellation point parallel
  end do
  !$omp end target parallel

  !$omp target parallel do
  do i = 1, N
    a = 3.14
    !ERROR: With PARALLEL clause, CANCELLATION POINT construct cannot be closely nested inside TARGET PARALLEL DO construct
    !$omp cancellation point parallel
  end do
  !$omp end target parallel do

  !$omp do
  do i = 1, N
    a = 3.14
    !ERROR: With PARALLEL clause, CANCELLATION POINT construct cannot be closely nested inside DO construct
    !$omp cancellation point parallel
  end do
  !$omp end do

contains
  subroutine sub1()
    !$omp task
    !$omp cancellation point taskgroup
    a = 3.14
    !$omp end task

    !$omp taskloop
    do i = 1, N
      !$omp parallel
      !$omp end parallel
      !$omp cancellation point taskgroup
      a = 3.14
    end do
    !$omp end taskloop

    !$omp taskloop nogroup
    do i = 1, N
      !$omp cancellation point taskgroup
      a = 3.14
    end do

    !$omp parallel
    !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
    !$omp cancellation point taskgroup
    a = 3.14
    !$omp end parallel

    !$omp do
    do i = 1, N
      !$omp task
      !$omp cancellation point taskgroup
      a = 3.14
      !$omp end task
    end do
    !$omp end do

    !$omp parallel
    !$omp taskgroup
    !$omp task
    !$omp cancellation point taskgroup
    a = 3.14
    !$omp end task
    !$omp end taskgroup
    !$omp end parallel

    !$omp parallel
    !$omp task
    !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
    !$omp cancellation point taskgroup
    a = 3.14
    !$omp end task
    !$omp end parallel

    !$omp parallel
    !$omp do
    do i = 1, N
      !$omp task
      !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
      !$omp cancellation point taskgroup
      a = 3.14
      !$omp end task
    end do
    !$omp end do
    !$omp end parallel

    !$omp target parallel
    !$omp task
    !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
    !$omp cancellation point taskgroup
    a = 3.14
    !$omp end task
    !$omp end target parallel

    !$omp parallel
    !$omp taskloop private(j) nogroup
    do i = 1, N
      !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
      !$omp cancellation point taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end parallel

    !$omp parallel
    !$omp taskloop
    do i = 1, N
      !$omp cancellation point taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end parallel

    !$omp parallel
    !$omp taskgroup
    !$omp taskloop nogroup
    do i = 1, N
      !$omp cancellation point taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end taskgroup
    !$omp end parallel

    !$omp target parallel
    !$omp taskloop nogroup
    do i = 1, N
      !ERROR: With TASKGROUP clause, CANCELLATION POINT construct must be closely nested inside TASK or TASKLOOP construct and CANCELLATION POINT region must be closely nested inside TASKGROUP region
      !$omp cancellation point taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end target parallel
  end subroutine sub1

end program main
