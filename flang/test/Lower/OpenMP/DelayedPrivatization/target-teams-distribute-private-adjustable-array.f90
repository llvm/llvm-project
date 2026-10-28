! Test GPU delayed privatization for target teams distribute parallel do:
! a dynamic-extent private array is heap-allocated (with cleanup), while a
! constant-size private array is privatized unboxed as a plain fir.array.

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir \
! RUN:     -fopenmp -fopenmp-is-target-device \
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

! A static (constant-size, trivial-element) private array is privatized unboxed
! as a plain fir.array -- no descriptor and no init region, even on a GPU target.
! The trailing end-of-line anchor confirms there is no `init {` region:
! CHECK: omp.private {type = private} @{{.*}}Etmp_private_64xf32 : !fir.array<64xf32>{{$}}

! CHECK-LABEL: omp.private {type = private} @{{.*}}Etmp_private_heap_box_Uxf32 : !fir.box<!fir.array<?xf32>> init {
! CHECK:         %[[DIMS:.*]]:3 = fir.box_dims
! CHECK:         fir.allocmem !fir.array<?xf32>, %[[DIMS]]#1
! CHECK:       } dealloc {
! CHECK:         fir.freemem
