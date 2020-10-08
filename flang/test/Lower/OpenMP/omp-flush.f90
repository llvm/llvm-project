! This test checks lowering of OpenMP Flush Directive.

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program flush

        integer :: a,b,c

!FIRDialect-LABEL:func @_QQmain() {
!FIRDialect:  %{{.*}} = fir.alloca i32 {name = "{{.*}}Ea"}
!FIRDialect:  %{{.*}} = fir.alloca i32 {name = "{{.*}}Eb"}
!FIRDialect:  %{{.*}} = fir.alloca i32 {name = "{{.*}}Ec"}

!LLVMIRDialect-LABEL: llvm.func @_QQmain() {
!LLVMIRDialect:   %{{.*}} = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "{{.*}}Ea"} : (!llvm.i64) -> !llvm.ptr<i32>
!LLVMIRDialect:   %{{.*}} = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "{{.*}}Eb"} : (!llvm.i64) -> !llvm.ptr<i32>
!LLVMIRDialect:   %{{.*}} = llvm.alloca %{{.*}} x !llvm.i32 {in_type = i32, name = "{{.*}}Ec"} : (!llvm.i64) -> !llvm.ptr<i32>

!LLVMIR-LABEL: define void @_QQmain() {{.*}} {
!LLVMIR:   %{{.*}} = alloca i32, i64 1, align 4
!LLVMIR:   %{{.*}} = alloca i32, i64 1, align 4
!LLVMIR:   %{{.*}} = alloca i32, i64 1, align 4
!LLVMIR:   call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.*}})
!LLVMIR:   br label %omp_parallel
!$OMP PARALLEL
!FIRDialect:  omp.parallel {

!LLVMIRDialect:   omp.parallel {

!LLVMIR-LABEL: define internal void @_QQmain..omp_par
!LLVMIR: call void @__kmpc_flush(%struct.ident_t* @{{.*}})
!$OMP FLUSH(a,b,c)
!$OMP FLUSH
!FIRDialect:      omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!FIRDialect:      omp.flush
!FIRDialect:      %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect:      %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect:      %{{.*}} = addi %{{.*}}, %{{.*}} : i32
!FIRDialect:      fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>

!LLVMIRDialect:     omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>)
!LLVMIRDialect:     omp.flush
!LLVMIRDialect:     %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect:     %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect:     %{{.*}} = llvm.add %{{.*}}, %{{.*}} : !llvm.i32
!LLVMIRDialect:     llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<i32>
        c = a + b
!FIRDialect:    omp.terminator
!FIRDialect:  }

!LLVMIRDialect:     omp.terminator
!LLVMIRDialect:   }
!$OMP END PARALLEL

!$OMP FLUSH(a,b,c)
!$OMP FLUSH
!FIRDialect:      omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!FIRDialect:      omp.flush
!FIRDialect:      %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect:      %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
!FIRDialect:      %{{.*}} = addi %{{.*}}, %{{.*}} : i32
!FIRDialect:      fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>

!LLVMIRDialect:     omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>)
!LLVMIRDialect:     omp.flush
!LLVMIRDialect:     %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect:     %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<i32>
!LLVMIRDialect:     %{{.*}} = llvm.add %{{.*}}, %{{.*}} : !llvm.i32
!LLVMIRDialect:     llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<i32>

        c = a + b
!$OMP FLUSH(a,b,c)
!$OMP FLUSH
!FIRDialect:      omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
!FIRDialect:      omp.flush

!LLVMIRDialect:     omp.flush(%{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<i32>, !llvm.ptr<i32>, !llvm.ptr<i32>)
!LLVMIRDialect:     omp.flush

        print*, "After Flushing"

end program
