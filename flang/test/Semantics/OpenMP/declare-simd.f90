! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m

!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: The name 'x' should refer to a procedure
!$omp declare_simd(x)

!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: DECLARE_SIMD directive should have at most one argument
!$omp declare_simd(f00, f01)

!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: The argument to the DECLARE_SIMD directive should be a procedure name
!$omp declare_simd(v : integer)

contains

subroutine f00
end

subroutine f01
end

integer function f02
!Ok, expect no errors
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!$omp declare_simd(f02)
end

end module
