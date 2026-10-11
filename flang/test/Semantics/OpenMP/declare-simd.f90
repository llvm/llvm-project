! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

module m

integer :: x

! A DECLARE_SIMD directive cannot appear in the specification part of a module.
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
!$omp declare_simd(f00)

contains

subroutine f00
!ERROR: The name 'x' should refer to a procedure
!$omp declare_simd(x)
end

subroutine f01
!ERROR: DECLARE_SIMD directive should have at most one argument
!$omp declare_simd(f00, f01)
end

subroutine f02
!ERROR: The argument to the DECLARE_SIMD directive should be a procedure name
!$omp declare_simd(v : integer)
end

integer function f03
!Ok, expect no diagnostics
!$omp declare_simd(f03)
end

end module
