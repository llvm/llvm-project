! This test checks lowering of OpenMP parallel Directive with
! `PROC_BIND` clause present with different values.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

subroutine procbind_clause()

!FIRDialect: omp.parallel proc_bind(master) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel proc_bind(master) {
!LLVMDialect:   omp.terminator
!LLVMialect: }

!! Value 2 denotes master.
!LLVMIR: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{.*}}, i32 %omp_global_thread_num, i32 2)
!$OMP PARALLEL PROC_BIND(MASTER)
!$OMP END PARALLEL

!FIRDialect: omp.parallel proc_bind(close) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel proc_bind(close) {
!LLVMDialect:   omp.terminator
!LLVMialect: }

!! Value 3 denotes close.
!LLVMIR: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 3)
!$OMP PARALLEL PROC_BIND(CLOSE)
!$OMP END PARALLEL

!FIRDialect: omp.parallel proc_bind(spread) {
!FIRDialect:   omp.terminator
!FIRDialect: }

!LLVMDialect: omp.parallel proc_bind(spread) {
!LLVMDialect:   omp.terminator
!LLVMialect: }

!! Value 4 denotes spread.
!LLVMIR: call void @__kmpc_push_proc_bind(%struct.ident_t* @{{.*}}, i32 %{{.*}}, i32 4)
!$OMP PARALLEL PROC_BIND(SPREAD)
!$OMP END PARALLEL

end subroutine
