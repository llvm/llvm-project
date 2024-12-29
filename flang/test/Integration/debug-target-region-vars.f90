! RUN: %flang_fc1 -fopenmp -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Test that variables inside OpenMP target region don't cause build failure.
subroutine test1
  implicit none
  real, allocatable :: xyz(:)
  integer :: i

  !$omp target simd map(from:xyz)
  do i = 1, size(xyz)
    xyz(i) = 5.0 * xyz(i)
  end do
end subroutine

subroutine test2 (xyz)
  integer :: i
  integer :: xyz(:)

  !$omp target map(from:xyz)
    !$omp do private(xyz)
      do i = 1, 10
        xyz(i) = i
      end do
  !$omp end target
end subroutine

!CHECK: DISubprogram(name: "test1"{{.*}})
!CHECK: DISubprogram(name: "test2"{{.*}})
