! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! 2.4 An array section designates a subset of the elements in an array. Although
! Substring shares similar syntax but cannot be treated as valid array section.

  character*8 c, b
  character a

  b = "HIFROMPGI"
  c = b(2:7)
  !ERROR: A substring cannot appear in a PRIVATE clause
  !$omp parallel private(c(1:3))
  a = c(1:1)
  !$omp end parallel
end
