!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(x)
  integer :: x(*)
  !ERROR: Whole assumed-size array 'x' may not appear here without subscripts
  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  !$omp taskgraph graph_reset(x)
  !$omp end taskgraph
end

subroutine f01
  !ERROR: At most one GRAPH_RESET clause can appear on the TASKGRAPH directive
  !$omp taskgraph graph_reset(.true.) graph_reset(.false.)
  !$omp end taskgraph
end
