! RUN: split-file %s %t
! RUN: %python %S/../test_errors.py %t/common.f90 %flang -fopenmp -fopenmp-version=51
! RUN: %python %S/../test_errors.py %t/common.f90 %flang -fopenmp -fopenmp-version=52
! RUN: not %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=51 %t/pure.f90 2>&1 | FileCheck %s --check-prefix=PURE51
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 %t/pure.f90
! RUN: %python %S/../test_errors.py %t/conditional.f90 %flang -fopenmp -fopenmp-version=60
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=51 %t/version.f90 2>&1 | FileCheck %s --check-prefix=VERSION
! RUN: %flang_fc1 -fsyntax-only -fopenmp -fopenmp-version=52 %t/version.f90 2>&1 | FileCheck %s --check-prefix=VERSION

! PURE51: The OpenMP directive 'ASSUME' is not allowed in a PURE procedure
! VERSION: An argument to NO_OPENMP is an OpenMP v6.0 feature, try -fopenmp-version=60

!--- common.f90
subroutine valid_assume_no_openmp(value)
  integer :: value
  !$omp assume no_openmp
  block
    value = 1
  end block
  !$omp assume no_openmp holds(value > 0)
    value = 2
  !$omp end assume
  !$omp assume holds(value > 0) no_openmp
  block
    !$omp assume no_openmp
      value = 3
    !$omp end assume
  end block
end subroutine valid_assume_no_openmp

subroutine duplicate_assume_no_openmp
  !ERROR: At most one NO_OPENMP clause can appear on ASSUME directive
  !$omp assume no_openmp no_openmp
  block
  end block
end subroutine duplicate_assume_no_openmp

!--- pure.f90
pure subroutine pure_assume_no_openmp(value)
  integer, intent(inout) :: value
  !$omp assume no_openmp
  block
    value = 4
  end block
end subroutine pure_assume_no_openmp

!--- conditional.f90
pure subroutine valid_conditional_assumptions(value)
  integer, intent(inout) :: value
  logical, parameter :: enabled = .not. .false._1
  !$omp assume no_openmp(enabled) holds(value > 0)
  block
    value = 1
  end block
  !$omp assume holds(value > 0) no_openmp(.false.)
    value = 2
  !$omp end assume
end subroutine valid_conditional_assumptions

subroutine invalid_conditional_assumptions(runtime_flag)
  logical :: runtime_flag
  !ERROR: Must be a constant value
  !$omp assume no_openmp(runtime_flag)
  block
  end block
  !ERROR: Must have LOGICAL type, but is INTEGER(4)
  !$omp assume no_openmp(1)
  block
  end block
  !ERROR: Must be a scalar value, but is a rank-1 array
  !$omp assume no_openmp([.true., .false.])
  block
  end block
  !ERROR: At most one NO_OPENMP clause can appear on ASSUME directive
  !$omp assume no_openmp(.true.) no_openmp(.false.)
  block
  end block
end subroutine invalid_conditional_assumptions

subroutine unsupported_conditional_assumes
  !ERROR: NO_OPENMP clause with an argument on ASSUMES directive is not yet supported
  !$omp assumes no_openmp(.false.)
end subroutine unsupported_conditional_assumes

!--- version.f90
subroutine conditional_version
  !$omp assume no_openmp(.false.)
  block
  end block
end subroutine conditional_version
