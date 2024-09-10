!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsb
subroutine sb(a)
  integer :: a(:)
!CHECK: omp.parallel
  !$omp parallel default(private)
!CHECK: hlfir.elemental
    if (any(a/=(/(100,i=1,5)/))) print *, "OK"
  !$omp end parallel
end subroutine

!CHECK-LABEL: func @_QPsb2
subroutine sb2()
  integer, parameter :: SIZE=20
  integer :: i, a(SIZE)

! Just check that the construct below doesn't hit a TODO in lowering.
!CHECK: omp.parallel
  !$omp parallel
    a = [ (i, i=1, SIZE) ]
    print *, i
  !$omp end parallel
end subroutine
