! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 2.18.1 Cancel Construct

program main
  integer :: i, N = 10
  real :: a

  !ERROR: CANCEL TASKGROUP directive is not closely nested inside TASK or TASKLOOP
  !$omp cancel taskgroup

  !ERROR: CANCEL SECTIONS directive is not closely nested inside SECTION or SECTIONS
  !$omp cancel sections

  !ERROR: CANCEL DO directive is not closely nested inside the construct that matches the DO clause type
  !$omp cancel do

  !ERROR: CANCEL PARALLEL directive is not closely nested inside the construct that matches the PARALLEL clause type
  !$omp cancel parallel

  !$omp parallel
  !$omp sections
  !$omp cancel sections
  !$omp section
  a = 3.14
  !$omp end sections
  !$omp end parallel

  !$omp sections
  !$omp section
  !$omp cancel sections
  a = 3.14
  !$omp end sections

  !$omp parallel
  !ERROR: With SECTIONS clause, CANCEL construct cannot be closely nested inside PARALLEL construct
  !$omp cancel sections
  a = 3.14
  !$omp end parallel

  !$omp parallel sections
  !$omp cancel sections
  a = 3.14
  !$omp end parallel sections

  !$omp do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end do

  !$omp parallel do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end parallel do

  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end distribute parallel do
  !$omp end teams
  !$omp end target

  !$omp target
  !$omp teams distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end teams distribute parallel do
  !$omp end target

  !$omp target teams distribute parallel do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end target teams distribute parallel do

  !$omp target parallel do
  do i = 1, N
    a = 3.14
    !$omp cancel do
  end do
  !$omp end target parallel do

  !$omp parallel
  do i = 1, N
    a = 3.14
    !ERROR: With DO clause, CANCEL construct cannot be closely nested inside PARALLEL construct
    !$omp cancel do
  end do
  !$omp end parallel

  !$omp parallel
  do i = 1, N
    a = 3.14
    !$omp cancel parallel
  end do
  !$omp end parallel

  !$omp target parallel
  do i = 1, N
    a = 3.14
    !$omp cancel parallel
  end do
  !$omp end target parallel

  !$omp target parallel do
  do i = 1, N
    a = 3.14
    !ERROR: With PARALLEL clause, CANCEL construct cannot be closely nested inside TARGET PARALLEL DO construct
    !$omp cancel parallel
  end do
  !$omp end target parallel do

  !$omp do
  do i = 1, N
    a = 3.14
    !ERROR: With PARALLEL clause, CANCEL construct cannot be closely nested inside DO construct
    !$omp cancel parallel
  end do
  !$omp end do

contains
  subroutine sub1()
    !$omp task
    !$omp cancel taskgroup
    a = 3.14
    !$omp end task

    !$omp taskloop
    do i = 1, N
      !$omp parallel
      !$omp end parallel
      !$omp cancel taskgroup
      a = 3.14
    end do
    !$omp end taskloop

    !$omp taskloop nogroup
    do i = 1, N
      !$omp cancel taskgroup
      a = 3.14
    end do

    !$omp parallel
    !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
    !$omp cancel taskgroup
    a = 3.14
    !$omp end parallel

    !$omp do
    do i = 1, N
      !$omp task
      !$omp cancel taskgroup
      a = 3.14
      !$omp end task
    end do
    !$omp end do

    !$omp parallel
    !$omp taskgroup
    !$omp task
    !$omp cancel taskgroup
    a = 3.14
    !$omp end task
    !$omp end taskgroup
    !$omp end parallel

    !$omp parallel
    !$omp task
    !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
    !$omp cancel taskgroup
    a = 3.14
    !$omp end task
    !$omp end parallel

    !$omp parallel
    !$omp do
    do i = 1, N
      !$omp task
      !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
      !$omp cancel taskgroup
      a = 3.14
      !$omp end task
    end do
    !$omp end do
    !$omp end parallel

    !$omp target parallel
    !$omp task
    !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
    !$omp cancel taskgroup
    a = 3.14
    !$omp end task
    !$omp end target parallel

    !$omp parallel
    !$omp taskloop private(j) nogroup
    do i = 1, N
      !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
      !$omp cancel taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end parallel

    !$omp parallel
    !$omp taskloop
    do i = 1, N
      !$omp cancel taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end parallel

    !$omp parallel
    !$omp taskgroup
    !$omp taskloop nogroup
    do i = 1, N
      !$omp cancel taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end taskgroup
    !$omp end parallel

    !$omp target parallel
    !$omp taskloop nogroup
    do i = 1, N
      !ERROR: With TASKGROUP clause, CANCEL construct must be closely nested inside TASK or TASKLOOP construct and CANCEL region must be closely nested inside TASKGROUP region
      !$omp cancel taskgroup
      a = 3.14
    end do
    !$omp end taskloop
    !$omp end target parallel
  end subroutine sub1

end program main
