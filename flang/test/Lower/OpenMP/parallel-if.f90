!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPtest1
subroutine test1(a)
integer :: a(:,:)
!CHECK: hlfir.destroy
!CHECK: omp.parallel if
!$omp parallel if(any(a .eq. 1))
!CHECK-NOT: hlfir.destroy
  print *, "Hello"
!$omp end parallel
end subroutine
