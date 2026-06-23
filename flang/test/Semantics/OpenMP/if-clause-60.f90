!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f
! Allowed in 6.0
  !$omp taskgraph if(.true.)
  !$omp end taskgraph

!ERROR: PARALLEL LOOP is not a constituent of the TEAMS LOOP directive
  !$omp teams loop if(parallel loop: .false.)
  do i = 1, 10
  end do
end
