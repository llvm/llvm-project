! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test the source code starting with omp syntax

! The directive is in the specification part of a main program, which is not a
! subroutine subprogram, function subprogram, or interface body.

!WARNING: DECLARE SIMD directive is not implemented yet and will be ignored
!ERROR: DECLARE SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
!$omp declare simd
integer :: x
end
