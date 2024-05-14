
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPsb1
subroutine sb1(arr)
  integer :: arr(:)
!CHECK: omp.parallel  {
  !$omp parallel
!CHECK: omp.single  {
  !$omp workshare
    arr = 0
  !$omp end workshare
!CHECK: }
  !$omp end parallel
!CHECK: }
end subroutine

!CHECK-LABEL: func @_QPsb2
subroutine sb2(arr)
  integer :: arr(:)
!CHECK: omp.parallel  {
  !$omp parallel
!CHECK: omp.single nowait {
  !$omp workshare
    arr = 0
  !$omp end workshare nowait
!CHECK: }
  !$omp end parallel
!CHECK: }
end subroutine

!CHECK-LABEL: func @_QPsb3
subroutine sb3(arr)
  integer :: arr(:)
!CHECK: omp.parallel  {
!CHECK: omp.single  {
  !$omp parallel workshare
    arr = 0
  !$omp end parallel workshare
!CHECK: }
!CHECK: }
end subroutine
