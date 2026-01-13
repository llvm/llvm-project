!REQUIRES: openmp_runtime
!RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp -fopenmp-version=60

module m
use omp_lib

implicit none
! Not in omp_lib yet.
integer, parameter :: omp_not_impex = 0
integer, parameter :: omp_import = 1
integer, parameter :: omp_export = 2
integer, parameter :: omp_impex = 3

contains

subroutine f00
  !$omp taskgraph
  !ERROR: Only task-generating constructs are allowed inside TASKGRAPH region
  !$omp parallel
  !$omp end parallel
  !$omp end taskgraph
end

subroutine f01
  !$omp taskgraph
  !$omp task
  !Non-task-generating constructs are ok if contained in an encountered task.
  !No diagnostic expected.
  !$omp parallel
  !$omp end parallel
  !$omp end task
  !$omp end taskgraph
end

subroutine f02
  !$omp taskgraph
  !ERROR: Transparent replayable tasks are not allowed in a TASKGRAPH region
  !$omp task transparent
  !$omp end task
  !$omp end taskgraph

  !$omp taskgraph
  !Not a transparent task.
  !No diagnostic expected.
  !$omp task transparent(omp_not_impex)
  !$omp end task
  !$omp end taskgraph

  !$omp taskgraph
  !Ok: transparent, but not replayable task.
  !No diagnostic expected.
  !$omp task replayable(.false.) transparent
  !$omp end task
  !$omp end taskgraph
end

subroutine f03
  integer(kind=omp_event_handle_kind) :: event

  !$omp taskgraph
  !ERROR: Detachable replayable tasks are not allowed in a TASKGRAPH region
  !$omp task detach(event)
  !$omp end task
  !$omp end taskgraph

  !$omp taskgraph
  !Ok: task is detachable, but not replayable.
  !No diagnostic expected
  !$omp task detach(event) replayable(.false.)
  !$omp end task
  !$omp end taskgraph
end

subroutine f04
  !$omp taskgraph
  !ERROR: Undeferred replayable tasks are not allowed in a TASKGRAPH region
  !$omp task if(.false.)
  !$omp end task
  !$omp end taskgraph

  !$omp taskgraph
  !Ok: task is undeferred, but not replayable.
  !No diagnostic expected.
  !$omp task if(.false.) replayable(.false.)
  !$omp end task
  !$omp end taskgraph
end

subroutine f05
  integer :: i

  !$omp taskgraph
  !ERROR: The NOGROUP clause must be specified on every construct in a TASKGRAPH region that could be enclosed in an implicit TASKGROUP
  !$omp taskloop
  do i = 1, 10
  enddo
  !$omp end taskloop
  !$omp end taskgraph

  !$omp taskgraph
  !This also applies to non-replayable constructs
  !ERROR: The NOGROUP clause must be specified on every construct in a TASKGRAPH region that could be enclosed in an implicit TASKGROUP
  !$omp taskloop replayable(.false.)
  do i = 1, 10
  enddo
  !$omp end taskloop
  !$omp end taskgraph

  !$omp taskgraph
  !No diagnostic expected.
  !$omp taskloop replayable(.false.) nogroup
  do i = 1, 10
  enddo
  !$omp end taskloop
  !$omp end taskgraph
end

end module
