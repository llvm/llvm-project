 ! RUN: %check_flang_tidy %s openmp-accumulator-race %t -- --extra-arg=-fopenmp
subroutine loop1(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel
  !$omp do
  do m = 1, N
     acc = acc + N
     ! CHECK-MESSAGES: :[[@LINE-1]]:6: warning: possible race condition on 'acc'
  end do
  !$omp end parallel
end subroutine loop1

subroutine loop2(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel do
  do m = 1, N
     !$omp critical
     acc = acc + N
     !$omp end critical
  end do
  !$omp end parallel do
end subroutine loop2

subroutine loop3(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel do
  do m = 1, N
     !$omp atomic update
     acc = acc + N
  end do
  !$omp end parallel do
end subroutine loop3

subroutine loop4(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel do private(acc)
  do m = 1, N
     acc = acc + N
  end do
  !$omp end parallel do
end subroutine loop4

subroutine loop5(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel do firstprivate(acc)
  do m = 1, N
     acc = acc + N
  end do
  !$omp end parallel do
end subroutine loop5

subroutine loop6(acc, N)
  integer, intent(inout) :: acc
  integer, intent(in) :: N
  integer :: m
  !$omp parallel do lastprivate(acc)
  do m = 1, N
     acc = acc + N
  end do
  !$omp end parallel do
end subroutine loop6
