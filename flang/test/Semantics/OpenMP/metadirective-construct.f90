!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! The CONSTRUCT trait set

subroutine f00
  !$omp metadirective &
!ERROR: CONDITION is not a valid trait for CONSTRUCT trait set
  !$omp & when(construct={condition(.true.)}: nothing)
end

subroutine f01
  !$omp metadirective &
!ERROR: Directive-name traits cannot have properties
  !$omp & when(construct={parallel(nowait), simd}: nothing)
end

subroutine f02
  !$omp metadirective &
!ERROR: SIMD trait requires a clause that is allowed on the DECLARE SIMD directive
  !$omp & when(construct={simd(nowait)}: nothing)
end

subroutine f03
  !$omp metadirective &
!ERROR: Extension traits are not valid for CONSTRUCT trait set
  !$omp & when(construct={fred(1)}: nothing)
end

subroutine f04
  !$omp metadirective &
!This is ok
  !$omp & when(construct={parallel, simd(simdlen(32), notinbranch)}: nothing)
end
