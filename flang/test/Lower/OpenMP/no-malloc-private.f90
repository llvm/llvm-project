! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device -o - %s | FileCheck %s

subroutine foo(state,ilast,jlast,vals)
  real, intent(in) :: state(:,:)
  integer, intent(in) :: ilast, jlast
  real, intent(  out) :: vals(:,:)
  
  real :: bar(4)
  integer :: i,k,ll,s

  !$omp target teams distribute parallel do private(bar)
  do i = 1, ilast
     do j = 1, jlast
        do s = 1, 4
           bar(s) = state(i,j+s)
        enddo
        vals(i,j) = -bar(1)/12 + 7*bar(2)/12 + 7*bar(3)/12 - bar(4)/12
     enddo
  enddo
  !$omp end target teams distribute parallel do
end subroutine foo

! Ensure that won't use heap allocations as part of the privatizer or anywhere
! inside of the function.

! CHECK: omp.private {type = private} @[[PRIVATIZER:.*]] : !fir.array<4xf32>
! CHECK-NOT: fir.allocmem
! CHECK: omp.yield

! CHECK-NOT: fir.allocmem
! CHECK: return
