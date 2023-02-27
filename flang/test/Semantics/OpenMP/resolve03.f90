! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! 2.15.3 Although variables in common blocks can be accessed by use association
! or host association, common block names cannot. As a result, a common block
! name specified in a data-sharing attribute clause must be declared to be a
! common block in the same scoping unit in which the data-sharing attribute
! clause appears.

  common /c/ a, b
  integer a(3), b

  A = 1
  B = 2
  block
    !ERROR: COMMON block must be declared in the same scoping unit in which the OpenMP directive or clause appears
    !$omp parallel shared(/c/)
    a(1:2) = 3
    B = 4
    !$omp end parallel
  end block
  print *, a, b
end
