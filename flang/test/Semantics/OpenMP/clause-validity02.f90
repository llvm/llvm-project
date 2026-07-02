! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45

subroutine bad_in_45(h_ptr)
  integer, pointer :: h_ptr
  !ERROR: USE_DEVICE_ADDR clause is not allowed on TARGET DATA directive in OpenMP v4.5, try -fopenmp-version=50
  !ERROR: At least one of MAP, USE_DEVICE_PTR clauses must appear on TARGET DATA directive
  !$omp target data use_device_addr(h_ptr)
  !$omp end target data
end

