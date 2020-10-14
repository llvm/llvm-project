! This test checks lowering of OpenMP parallel Directive with
! `DEFAULT` clause present with different values.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMDialect

subroutine default_clause()

!FIRDialect: omp.parallel default(private) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel default(private) {
!LLVMDialect:   omp.terminator
!LLVMialect: }
!$OMP PARALLEL DEFAULT(PRIVATE)
!$OMP END PARALLEL

!FIRDialect: omp.parallel default(firstprivate) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel default(firstprivate) {
!LLVMDialect:   omp.terminator
!LLVMialect: }
!$OMP PARALLEL DEFAULT(FIRSTPRIVATE)
!$OMP END PARALLEL

!FIRDialect: omp.parallel default(shared) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel default(shared) {
!LLVMDialect:   omp.terminator
!LLVMialect: }
!$OMP PARALLEL DEFAULT(SHARED)
!$OMP END PARALLEL

!FIRDialect: omp.parallel default(none) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel default(none) {
!LLVMDialect:   omp.terminator
!LLVMialect: }
!$OMP PARALLEL DEFAULT(NONE)
!$OMP END PARALLEL

end subroutine
