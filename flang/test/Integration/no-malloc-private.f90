! RUN: %flang_fc1 -emit-llvm -fopenmp -o - -x f95 %s | FileCheck %s

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

! Ensure that we do not generate a call to malloc
!CHECK-LABEL: omp.private.init:
!CHECK-NOT:   call {{.*}} @malloc
!CHECK:       br label

