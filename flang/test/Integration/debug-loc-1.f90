!RUN: %flang_fc1 -emit-llvm -debug-info-kind=line-tables-only -fopenmp %s -o - | FileCheck %s

! Test that this file builds without an error.

module debugloc
contains
subroutine test1
implicit none
 integer :: i
 real, save :: var

! CHECK: DILocation(line: [[@LINE+1]], {{.*}})
!$omp parallel do
do i=1,100
  var = var + 0.1
end do
!$omp end parallel do

end subroutine test1

subroutine test2

real, save :: tp
!$omp threadprivate (tp)
! CHECK: DILocation(line: [[@LINE+1]], {{.*}})
  tp = tp + 1

end subroutine test2

end module debugloc
