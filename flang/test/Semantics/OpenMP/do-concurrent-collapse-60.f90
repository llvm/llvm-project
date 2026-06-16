!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f
  integer :: i

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp parallel do collapse(2)
  do i = 1, 1
    !BECAUSE: DO CONCURRENT loop is not a valid affected loop
    do concurrent (integer :: j = 1:2)
      print *, j
    end do
  end do

  !BECAUSE: ORDERED clause was specified with argument 2
  !$omp parallel do ordered(2)
  !ERROR: DO CONCURRENT must be the only affected loop in a loop nest
  do concurrent (integer :: j = 1:2)
    do i = 1, 2
      print *, i
    end do
  end do

  !Ok, DO CONCURRENT is not an affected loop
  !$omp parallel do
  do i = 1, 1
    do concurrent (integer :: j = 1:2)
      print *, j
    end do
  end do

  !Ok, DO CONCURRENT is the only affected loop
  !$omp parallel do
  do concurrent (integer :: j = 1:2)
    print *, j
  end do

  !Ok, DO CONCURRENT is the only affected loop
  !$omp loop
  do concurrent (integer :: j = 1:2)
    print *, j
  end do

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp loop collapse(2)
  do i = 1, 1
    !BECAUSE: DO CONCURRENT loop is not a valid affected loop
    do concurrent (integer :: j = 1:2)
      print *, j
    end do
  end do
end
