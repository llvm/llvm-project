! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenmp %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenMP declarative construct

! CHECK: Module m
module m
  real, dimension(10) :: x
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(x)
end
! CHECK: End Module m

! CHECK: Module m2
module m2
  integer, save :: i
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(i)
contains
  subroutine sub()
    i = 1;
  end
  subroutine sub2()
    i = 2;
  end
end
! CHECK: End Module m2

! CHECK: Program main
program main
  real :: y
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(y)
end
! CHECK: End Program main

! CHECK: Subroutine sub1
subroutine sub1()
  real, save :: p
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(p)
end
! CHECK: End Subroutine sub1

! CHECK: Subroutine sub2
subroutine sub2()
  real, save :: q
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(q)
contains
  subroutine sub()
  end
end
! CHECK: End Subroutine sub2

