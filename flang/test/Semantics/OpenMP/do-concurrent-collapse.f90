!RUN: %python %S/../test_errors.py %s %flang -fopenmp

integer :: i, j
! ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
! BECAUSE: COLLAPSE clause was specified with argument 2
! ERROR: DO CONCURRENT loops cannot be used with the COLLAPSE clause.
!$omp parallel do collapse(2)
do i = 1, 1
  ! BECAUSE: This is not a valid intervening code
  ! ERROR: DO CONCURRENT loops cannot form part of a loop nest.
  do concurrent (j = 1:2)
    print *, j
  end do
end do

!$omp parallel do
do i = 1, 1
  ! This should not lead to an error because it is not part of a loop nest:
  do concurrent (j = 1:2)
    print *, j
  end do
end do

!$omp parallel do
! ERROR: DO CONCURRENT loops cannot form part of a loop nest.
do concurrent (j = 1:2)
  print *, j
end do

!$omp loop
! Do concurrent is explicitly allowed inside of omp loop
do concurrent (j = 1:2)
  print *, j
end do

! ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
! BECAUSE: COLLAPSE clause was specified with argument 2
! ERROR: DO CONCURRENT loops cannot be used with the COLLAPSE clause.
!$omp loop collapse(2)
do i = 1, 1
  ! BECAUSE: This is not a valid intervening code
  do concurrent (j = 1:2)
    print *, j
  end do
end do
end
