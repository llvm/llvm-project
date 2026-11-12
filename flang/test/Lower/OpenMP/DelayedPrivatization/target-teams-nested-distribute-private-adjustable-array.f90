! Test that GPU delayed privatization allocates dynamic private arrays for
! standalone distribute on the heap and emits cleanup.

! RUN: %if amdgpu-registered-target %{ \
! RUN:   %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir \
! RUN:     -fopenmp -fopenmp-is-target-device \
! RUN:     -o - %s 2>&1 | FileCheck %s \
! RUN: %}

subroutine nested_distribute_private_tmp(b)
  real, dimension(:,:), intent(inout) :: b
  real, dimension(size(b, 1)) :: tmp
  integer :: i, j

  !$omp target teams map(tofrom:b)
    !$omp distribute private(tmp)
    do i = 1, 1
      do j = 1, size(b, 1)
        tmp(j) = b(j, i)
      end do
      b(1, i) = tmp(1)
    end do
  !$omp end target teams
end subroutine

! CHECK: warning: {{.*}}OpenMP private dynamic array 'tmp' on a GPU target may exceed stack memory; using device heap allocation instead, which can severely degrade performance

! CHECK-LABEL: omp.private {type = private} @{{.*}}Etmp_private_heap_box_Uxf32 : !fir.box<!fir.array<?xf32>> init {
! CHECK:         %[[DIMS:.*]]:3 = fir.box_dims
! CHECK:         fir.allocmem !fir.array<?xf32>, %[[DIMS]]#1
! CHECK:       } dealloc {
! CHECK:         fir.freemem
