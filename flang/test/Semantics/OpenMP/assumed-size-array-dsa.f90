!RUN: %python %S/../test_errors.py %s %flang -fopenmp

! Assumed-size arrays are predetermined shared and may only appear in a SHARED clause
subroutine test_assumed_size_array_dsa( arr, N )
  implicit none
  integer :: arr(*)
  integer :: i, N

  !$omp parallel

  !$omp task shared(arr)
  do i = 1, N
    print *, arr(i)
  end do
  !$omp end task

  ! ERROR: Assumed-size array 'arr' may not appear in a PRIVATE clause
  !$omp task private(arr)
  do i = 1, N
    print *, arr(i)
  end do
  !$omp end task

  ! ERROR: Assumed-size array 'arr' may not appear in a FIRSTPRIVATE clause
  !$omp task firstprivate(arr)
  do i = 1, N
    print *, arr(i)
  end do
  !$omp end task

  ! ERROR: Assumed-size array 'arr' may not appear in a LASTPRIVATE clause
  !$omp do lastprivate(arr)
  do i = 1, N
    print *, arr(i)
  end do
  !$omp end do

  ! ERROR: Assumed-size array 'arr' may not appear in a LINEAR clause
  ! ERROR: List item 'arr' in LINEAR clause must be a scalar variable
  !$omp do linear(arr)
  do i = 1, N
    print *, arr(i)
  end do
  !$omp end do

  !$omp end parallel

end subroutine
