!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00(x)
  logical :: x
  !WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
  !ERROR: An argument to REVERSE_OFFLOAD is an OpenMP v6.0 feature, try -fopenmp-version=60
  !ERROR: Must be a constant value
  !$omp requires reverse_offload(x)
end

subroutine f01
  !WARNING: REVERSE_OFFLOAD clause is not supported and will be ignored
  !WARNING: An argument to REVERSE_OFFLOAD is an OpenMP v6.0 feature, try -fopenmp-version=60
  !$omp requires reverse_offload(.true.)
end
