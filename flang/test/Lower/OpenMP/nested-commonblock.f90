! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_nested()
!CHECK:         omp.parallel private(@_QFtest_nestedEa_firstprivate_i32 {{.*}}, @_QFtest_nestedEb_firstprivate_i32 {{.*}})
!CHECK:           omp.parallel private(@_QFtest_nestedEa_private_i32 {{.*}}, @_QFtest_nestedEb_private_i32 {{.*}})

subroutine test_nested()
  integer :: a, b
  common /com/ a, b

  !$omp parallel firstprivate(/com/)
    !$omp parallel private(/com/)
    !$omp end parallel
  !$omp end parallel
end subroutine
