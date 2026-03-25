!RUN: %python %S/../test_errors.py %s %flang -fopenmp

integer :: i, j
! ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
! BECAUSE: COLLAPSE clause was specified with argument 2
! ERROR: DO CONCURRENT loops cannot be used with the COLLAPSE clause.
!$omp parallel do collapse(2)
do i = 1, 1
  ! BECAUSE: DO CONCURRENT loop is not a valid affected loop
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

! ERROR: This construct requires a canonical loop nest
!$omp parallel do
! BECAUSE: DO CONCURRENT loop is not a valid affected loop
do concurrent (j = 1:2)
  print *, j
end do

! ERROR: This construct requires a canonical loop nest
!$omp loop
! BECAUSE: DO CONCURRENT loop is not a valid affected loop
do concurrent (j = 1:2)
  print *, j
end do

! ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
! BECAUSE: COLLAPSE clause was specified with argument 2
! ERROR: DO CONCURRENT loops cannot be used with the COLLAPSE clause.
!$omp loop collapse(2)
do i = 1, 1
  ! BECAUSE: DO CONCURRENT loop is not a valid affected loop
  do concurrent (j = 1:2)
    print *, j
  end do
end do
end
