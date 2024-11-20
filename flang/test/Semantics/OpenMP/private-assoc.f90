! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! The ASSOCIATE name preserves the association with the selector established
! in the associate statement. Therefore it is incorrect to change the
! data-sharing attribute of the name.

subroutine assoc_private(x)
  integer :: x
  associate(z => x)
  !ERROR: Variable 'z' in ASSOCIATE cannot be in a PRIVATE clause
  !$omp parallel private(z)
  !$omp end parallel
  end associate
end subroutine

subroutine assoc_firstprivate(x)
  integer :: x
  associate(z => x)
  !ERROR: Variable 'z' in ASSOCIATE cannot be in a FIRSTPRIVATE clause
  !$omp parallel firstprivate(z)
  !$omp end parallel
  end associate
end subroutine

subroutine assoc_lastprivate(x)
  integer :: x
  associate(z => x)
  !ERROR: Variable 'z' in ASSOCIATE cannot be in a LASTPRIVATE clause
  !$omp parallel sections lastprivate(z)
  !$omp end parallel sections
  end associate
end subroutine
