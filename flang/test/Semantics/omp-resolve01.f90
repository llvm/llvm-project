! RUN: %S/test_errors.sh %s %t %flang -fopenmp

! 2.4 An array section designates a subset of the elements in an array. Although
! Substring shares similar syntax but cannot be treated as valid array section.

  character*8 c, b
  character a

  b = "HIFROMPGI"
  c = b(2:7)
  !ERROR: Substrings are not allowed on OpenMP directives or clauses
  !$omp parallel private(c(1:3))
  a = c(1:1)
  !$omp end parallel
end
