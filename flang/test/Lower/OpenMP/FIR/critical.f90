!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefix="OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | tco | FileCheck %s --check-prefix="LLVMIR"

!OMPDialect: omp.critical.declare @help2 hint(none)
!OMPDialect: omp.critical.declare @help1 hint(contended)

subroutine omp_critical()
  use omp_lib
  integer :: x, y
!OMPDialect: omp.critical(@help1)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help1.var, i32 2)
!$OMP CRITICAL(help1) HINT(omp_lock_hint_contended)
  x = x + y
!OMPDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help1.var)
!$OMP END CRITICAL(help1)

! Test that the same name can be used again
! Also test with the zero hint expression
!OMPDialect: omp.critical(@help2)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help2.var, i32 0)
!$OMP CRITICAL(help2) HINT(omp_lock_hint_none)
  x = x - y
!OMPDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help2.var)
!$OMP END CRITICAL(help2)

!OMPDialect: omp.critical
!LLVMIR: call void @__kmpc_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP CRITICAL
  y = x + y
!OMPDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP END CRITICAL
end subroutine omp_critical
