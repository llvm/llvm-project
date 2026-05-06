!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00(x)
  logical :: x
  !ERROR: An argument to REVERSE_OFFLOAD is an OpenMP v6.0 feature, try -fopenmp-version=60
  !ERROR: Must be a constant value
  !$omp requires reverse_offload(x)
end

subroutine f01
  !WARNING: An argument to REVERSE_OFFLOAD is an OpenMP v6.0 feature, try -fopenmp-version=60
  !$omp requires reverse_offload(.true.)
end
