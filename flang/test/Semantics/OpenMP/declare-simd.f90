! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m

!ERROR: The name 'x' should refer to a procedure
!$omp declare_simd(x)

!ERROR: DECLARE_SIMD directive should have at most one argument
!$omp declare_simd(f00, f01)

!ERROR: The argument to the DECLARE_SIMD directive should be a procedure name
!$omp declare_simd(v : integer)

contains

subroutine f00
end

subroutine f01
end

integer function f02
!Ok, expect no diagnostics
!$omp declare_simd(f02)
end

end module
