!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: x, y, v

  !ERROR: ATOMIC UPDATE operation with CAPTURE should contain two statements
  !$omp atomic update capture
  x = v
  x = x + 1
  y = x
  !$omp end atomic
end

subroutine f01
  integer :: x, y, v

  !ERROR: ATOMIC UPDATE operation with CAPTURE should contain two assignments
  !$omp atomic update capture
  x = v
  block
    x = x + 1
    y = x
  end block
  !$omp end atomic
end

subroutine f02
  integer :: x, y

  ! The update and capture statements can be inside of a single BLOCK.
  ! The end-directive is then optional. Expect no diagnostics.
  !$omp atomic update capture
  block
    x = x + 1
    y = x
  end block
end

subroutine f03
  integer :: x

  !ERROR: Unable to identify capture statement: in ATOMIC UPDATE operation with CAPTURE the source value in the capture statement should be a variable (with the same type as the target)
  !$omp atomic update capture
  x = x + 1
  x = x + 2
  !$omp end atomic
end

subroutine f04
  integer :: x, v

  !$omp atomic update capture
  !WARNING: In ATOMIC UPDATE operation with CAPTURE either statement could be the capture and the update, assuming the first one is the capture statement
  v = x
  x = v
  !$omp end atomic
end

subroutine f05
  integer :: x, v, z

  !$omp atomic update capture
  v = x
  !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to x
  z = x + 1
  !$omp end atomic
end

subroutine f06
  integer :: x, v, z

  !$omp atomic update capture
  !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to x
  z = x + 1
  v = x
  !$omp end atomic
end
