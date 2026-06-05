! Test that GPU delayed privatization does not force heap allocation for
! adjustable arrays privatized on target teams distribute parallel do simd.

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -mmlir --enable-delayed-privatization-staging \
! RUN:     -o - %s 2>&1 | FileCheck %s \
! RUN: %}

subroutine dynamic_private_tmp_simd(b)
  real, dimension(:,:), intent(inout) :: b
  real, dimension(size(b, 1)) :: tmp
  integer :: i, j, k

  !$omp target teams distribute parallel do simd collapse(2) private(tmp)
  do j = 1, 1
    do i = 1, 1
      do k = 1, size(b, 1)
        tmp(k) = 1.0
      end do
      b(i,j) = tmp(1)
    end do
  end do
end subroutine

! CHECK-NOT: warning: OpenMP private dynamic array

! CHECK-LABEL: omp.private {type = private} @{{.*}}Etmp_private_box_Uxf32 : !fir.box<!fir.array<?xf32>> init {
! CHECK-NOT:     fir.allocmem
! CHECK:         fir.alloca !fir.array<?xf32>
! CHECK:         omp.yield
