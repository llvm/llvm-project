! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=50

use omp_lib
  implicit none
  integer :: xyz, abc
  real :: reduction_var
  !$omp parallel num_threads(4)
    !$omp single
      print *, "The"
    !$omp taskgroup
      !$omp task
      print *, "almighty"
      !$omp end task
      !$omp task
      print *, "sun"
      !$omp end task
    !$omp end taskgroup
    !$omp end single
  !$omp end parallel

  !$omp parallel private(xyz)
    !$omp taskgroup allocate(xyz)
      !$omp task
        print *, "The "
        !$omp taskgroup allocate(omp_large_cap_mem_space: abc)
          !$omp task
          print *, "almighty sun"
          !$omp end task
        !$omp end taskgroup
      !$omp end task
    !$omp end taskgroup
  !$omp end parallel

  !ERROR: PRIVATE clause is not allowed on the TASKGROUP directive
  !$omp taskgroup private(abc)
  !$omp end taskgroup

  !$omp parallel
    !$omp task
      !$omp taskgroup task_reduction(+ : reduction_var)
          print *, "The "
        !$omp taskgroup task_reduction(.or. : reduction_var) task_reduction(.and. : reduction_var)
          print *, "almighty sun"
        !$omp end taskgroup
      !$omp end taskgroup
    !$omp end task
  !$omp end parallel
end program
