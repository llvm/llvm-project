! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s
program main
!CHECK-LABEL:  MainProgram scope: main
  integer, parameter :: n = 256
  real(8) :: a(256)
  !$omp target map(mapper(xx), from:a)
  do i=1,n
     a(i) = 4.2
  end do
  !$omp end target
!CHECK:    OtherConstruct scope: size=0 alignment=1 sourceRange=74 bytes
!CHECK:    OtherClause scope: size=0 alignment=1 sourceRange=0 bytes
!CHECK:    xx: Misc ConstructName
end program main
