! Test that GPU delayed privatization allocates dynamic private arrays for
! target teams distribute parallel do on the heap and emits cleanup.

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -mmlir --enable-delayed-privatization-staging \
! RUN:     -o - %s 2>&1 | FileCheck %s \
! RUN: %}

subroutine dynamic_private_tmp(b)
  real, dimension(:,:), intent(inout) :: b
  real, dimension(size(b, 1)) :: tmp
  integer :: i, j, k

  !$omp target teams distribute parallel do collapse(2) private(tmp)
  do j = 1, 1
    do i = 1, 1
      do k = 1, size(b, 1)
        tmp(k) = 1.0
      end do
      b(i,j) = tmp(1)
    end do
  end do
end subroutine

subroutine static_private_tmp(b)
  real, dimension(:,:), intent(inout) :: b
  real, dimension(64) :: tmp
  integer :: i, j, k

  !$omp target teams distribute parallel do collapse(2) private(tmp)
  do j = 1, 1
    do i = 1, 1
      do k = 1, 64
        tmp(k) = 1.0
      end do
      b(i,j) = tmp(1)
    end do
  end do
end subroutine

! CHECK: warning: {{.*}}OpenMP private dynamic array 'tmp' on a GPU target may exceed stack memory; using device heap allocation instead, which can severely degrade performance

! CHECK-LABEL: omp.private {type = private} @{{.*}}Etmp_private_box_64xf32 : !fir.box<!fir.array<64xf32>> init {
! CHECK-NOT:     fir.allocmem
! CHECK:         fir.alloca !fir.array<64xf32>
! CHECK:         omp.yield

! CHECK-LABEL: omp.private {type = private} @{{.*}}Etmp_private_heap_box_Uxf32 : !fir.box<!fir.array<?xf32>> init {
! CHECK:         %[[DIMS:.*]]:3 = fir.box_dims
! CHECK:         fir.allocmem !fir.array<?xf32>, %[[DIMS]]#1
! CHECK:       } dealloc {
! CHECK:         fir.freemem
