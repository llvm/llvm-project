! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m

integer :: x

! A DECLARE_SIMD directive cannot appear in the specification part of a module.
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
!$omp declare_simd(f00)

contains

subroutine f00
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: The name 'x' should refer to a procedure
!$omp declare_simd(x)
end

subroutine f01
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: DECLARE_SIMD directive should have at most one argument
!$omp declare_simd(f00, f01)
end

subroutine f02
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!ERROR: The argument to the DECLARE_SIMD directive should be a procedure name
!$omp declare_simd(v : integer)
end

integer function f03
!Ok, expect no errors
!WARNING: DECLARE_SIMD directive is not implemented yet and will be ignored
!$omp declare_simd(f03)
end

end module
