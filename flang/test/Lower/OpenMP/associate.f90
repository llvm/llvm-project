! Check that constructs with associate and variables that have implicitly
! determined DSAs are lowered properly.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_parallel_assoc
!CHECK:         omp.parallel {
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEa"}
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEb"}
!CHECK:           omp.wsloop {
!CHECK:           }
!CHECK:         }
!CHECK:         omp.parallel {{.*}} {
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEb"}
!CHECK:           omp.wsloop {
!CHECK:           }
!CHECK:         }
subroutine test_parallel_assoc()
  integer, parameter :: l = 3
  integer :: a(l)
  integer :: i
  a = 1

  !$omp parallel do
  do i = 1,l
    associate (b=>a)
      b(i) = b(i) * 2
    end associate
  enddo
  !$omp end parallel do

  !$omp parallel do default(private)
  do i = 1,l
    associate (b=>a)
      b(i) = b(i) * 2
    end associate
  enddo
  !$omp end parallel do
end subroutine
