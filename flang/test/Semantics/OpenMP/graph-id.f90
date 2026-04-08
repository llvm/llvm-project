!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=8_8)
  !$omp taskgraph graph_id("my graph")
  !$omp end taskgraph
end

subroutine f01
  !ERROR: At most one GRAPH_ID clause can appear on the TASKGRAPH directive
  !$omp taskgraph graph_id(1) graph_id(2)
  !$omp end taskgraph
end
