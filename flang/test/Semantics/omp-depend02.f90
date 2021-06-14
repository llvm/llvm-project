! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.13.9 Depend Clause
! A variable that is part of another variable
! (such as an element of a structure) but is not an array element or
! an array section cannot appear in a DEPEND clause

subroutine vec_mult(N)
  implicit none
  integer :: i, N
  real, allocatable :: p(:), v1(:), v2(:)

  type my_type
    integer :: a(10)
  end type my_type

  type(my_type) :: my_var
  allocate( p(N), v1(N), v2(N) )

  !$omp parallel num_threads(2)
  !$omp single

  !$omp task depend(out:v1)
  call init(v1, N)
  !$omp end task

  !$omp task depend(out:v2)
  call init(v2, N)
  !$omp end task

  !ERROR: A variable that is part of another variable (such as an element of a structure) but is not an array element or an array section cannot appear in a DEPEND clause
  !$omp target nowait depend(in:v1,v2, my_var%a) depend(out:p) &
  !$omp& map(to:v1,v2) map(from: p)
  !$omp parallel do
  do i=1,N
    p(i) = v1(i) * v2(i)
  end do
  !$omp end target

  !$omp task depend(in:p)
  call output(p, N)
  !$omp end task

  !$omp end single
  !$omp end parallel

  deallocate( p, v1, v2 )

end subroutine
