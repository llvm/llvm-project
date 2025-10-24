! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s
program main
!CHECK-LABEL:  MainProgram scope: MAIN
  type ty
    real(4) :: x
  end type ty
  !$omp declare mapper(xx : ty :: v) map(v)
  integer, parameter :: n = 256
  type(ty) :: a(256)
  !$omp target map(mapper(xx), from:a)
  do i=1,n
     a(i)%x = 4.2
  end do
  !$omp end target
!CHECK:    xx: MapperDetails
end program main
