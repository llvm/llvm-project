!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefix="FIRDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefix="LLVMDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --fir-to-llvm-ir | tco | FileCheck %s --check-prefix="LLVMIR"

subroutine omp_critical()
  use omp_lib
  integer :: x, y
!FIRDialect: omp.critical.declare @help hint(contended)
!LLVMDialect: omp.critical.declare @help hint(contended)
!FIRDialect: omp.critical(@help)
!LLVMDialect: omp.critical(@help)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var, i32 2)
!$OMP CRITICAL(help) HINT(omp_lock_hint_contended)
  x = x + y
!FIRDialect: omp.terminator
!LLVMDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var)
!$OMP END CRITICAL(help)

! Test that the same name can be used again
! Also test with the zero hint expression
!FIRDialect: omp.critical(@help)
!LLVMDialect: omp.critical(@help)
!LLVMIR: call void @__kmpc_critical_with_hint({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var, i32 2)
!$OMP CRITICAL(help) HINT(omp_lock_hint_none)
  x = x - y
!FIRDialect: omp.terminator
!LLVMDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}help.var)
!$OMP END CRITICAL(help)

!FIRDialect: omp.critical
!LLVMDialect: omp.critical
!LLVMIR: call void @__kmpc_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP CRITICAL
  y = x + y
!FIRDialect: omp.terminator
!LLVMDialect: omp.terminator
!LLVMIR: call void @__kmpc_end_critical({{.*}}, {{.*}}, {{.*}} @{{.*}}_.var)
!$OMP END CRITICAL
end subroutine omp_critical
