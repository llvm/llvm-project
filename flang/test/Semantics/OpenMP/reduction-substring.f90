! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=45
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

subroutine scalar_substring(c)
  character(4) :: c
  !$omp declare reduction(r:character(2),character(4):omp_out=omp_in)
  !ERROR: A substring cannot appear in a REDUCTION clause
  !$omp parallel reduction(r:c(2:3))
  !$omp end parallel
end

subroutine element_substring(c)
  character(4) :: c(2)
  !$omp declare reduction(r:character(2),character(4):omp_out=omp_in)
  !ERROR: A substring cannot appear in a REDUCTION clause
  !$omp parallel reduction(r:c(1)(2:3))
  !$omp end parallel
end

subroutine component_substring()
  type box
    character(4) :: field
  end type
  type(box) :: a
  !$omp declare reduction(r:type(box):omp_out=omp_in)
  !ERROR: A substring cannot appear in a REDUCTION clause
  !$omp parallel reduction(r:a%field(2:3))
  !$omp end parallel
end

subroutine target_substring(c)
  character(4) :: c
  !$omp declare reduction(r:character(2),character(4):omp_out=omp_in)
  !ERROR: A substring cannot appear in a REDUCTION clause
  !$omp target parallel reduction(r:c(2:3))
  !$omp end target parallel
end

subroutine whole_character(c)
  character(4) :: c
  !$omp declare reduction(r:character(4):omp_out=omp_in)
  !$omp parallel reduction(r:c)
  !$omp end parallel
end

subroutine complex_part(z)
  complex :: z
  !$omp parallel reduction(+:z%re)
  !$omp end parallel
end
