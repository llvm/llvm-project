! This test checks lowering of OpenMP DO Directive(Worksharing).

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect
! RUN: bbc -fopenmp -emit-llvm %s -o - | \
! RUN:   FileCheck %s --check-prefix=LLVMIRDialect
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program wsloop
        integer :: i
!FIRDialect: func @_QQmain()
!FIRDialect: %[[I:.*]] = fir.alloca i32 {name = "_QEi"}
!LLVMIRDialect: func @_QQmain()

!LLVMIR: define void @_QQmain()
!LLVMIR:call i32 @__kmpc_global_thread_num{{.*}}
!LLVMIR:  br label %omp_parallel

!$OMP PARALLEL
!FIRDialect-LABLEL:  omp.parallel {
!LLVMIRDialect-LABLEL:  omp.parallel {

!LLVMIR: omp_parallel:                                     ; preds = %0
!LLVMIR:   @__kmpc_fork_call
!$OMP DO SCHEDULE(static)
!FIRDialect:     %[[WS_LB:.*]] = constant 1 : i32
!FIRDialect:     %[[WS_UB:.*]] = constant 9 : i32
!FIRDialect:     %[[WS_STEP:.*]] = constant 1 : i32
!FIRDialect:    "omp.wsloop"(%[[WS_LB]], %[[WS_UB]], %[[WS_STEP]]) ( {

!LLVMIRDialect: "omp.wsloop"(%{{.*}}, %{{.*}}, %{{.*}}) ( {

!LLVMIR:  define internal void @_QQmain..omp_par
!LLVMIR:  omp.par.entry:
!LLVMIR:    br label %omp.par.region
!LLVMIR:  omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
!LLVMIR:    ret void
!LLVMIR:  omp.par.region:                                   ; preds = %omp.par.entry
!LLVMIR:    br label %omp.par.region1
!LLVMIR:  omp.par.region1:                                  ; preds = %omp.par.region
!LLVMIR:    br label %omp_loop.preheader
!LLVMIR:  omp_loop.preheader:                               ; preds = %omp.par.region1
!LLVMIR:    @__kmpc_global_thread_num
!LLVMIR:    @__kmpc_for_static_init_4u
!LLVMIR:    br label %omp_loop.header
!LLVMIR:  omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
!LLVMIR:    %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]

do i=1, 9
!FIRDialect:    fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} -> index {
print*, i
!FIRDialect:    %[[RTBEGIN:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect:    %[[LD_I:.*]] = fir.load %[[I]] : !fir.ref<i32>
!FIRDialect:    %[[CONVERTED:.*]] = fir.convert %[[LD_I]] : (i32) -> i64
!FIRDialect:    fir.call @_FortranAioOutputInteger64(%[[RTBEGIN]], %[[CONVERTED]]) : (!fir.ref<i8>, i64) -> i1
!FIRDialect:    fir.call @_FortranAioEndIoStatement(%[[RTBEGIN]]) : (!fir.ref<i8>) -> i32
!FIRDialect:  }

!LLVMIRDialect:  ^bb0(%arg0: i64):  // no predecessors
!LLVMIRDialect:    llvm.br ^bb1(%{{.*}}, %{{.*}} : i64, i64)
!LLVMIRDialect:  ^bb1(%{{.*}}: i64, %{{.*}}: i64):  // 2 preds: ^bb0, ^bb2
!LLVMIRDialect:    llvm.cond_br %{{.*}}, ^bb2, ^bb3
!LLVMIRDialect:  ^bb2:  // pred: ^bb1
!LLVMIRDialect:     llvm.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, !llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
!LLVMIRDialect:     llvm.call @_FortranAioOutputInteger64(%{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i64) -> i1
!LLVMIRDialect:     llvm.call @_FortranAioEndIoStatement(%{{.*}}) : (!llvm.ptr<i8>) -> i32
!LLVMIRDialect:    llvm.br ^bb1(%{{.*}}, %{{.*}} : i64, i64)
!LLVMIRDialect:  ^bb3:  // pred: ^bb1

!LLVMIR:  omp_loop.cond:                                    ; preds = %omp_loop.header
!LLVMIR:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
!LLVMIR:  omp_loop.exit:                                    ; preds = %omp_loop.cond
!LLVMIR:    @__kmpc_for_static_fini
!LLVMIR:    br label %omp_loop.after
!LLVMIR:  omp_loop.after:                                   ; preds = %omp_loop.exit
!LLVMIR:    br label %omp.par.pre_finalize
!LLVMIR:  omp.par.pre_finalize:                             ; preds = %omp_loop.after
!LLVMIR:    br label %omp.par.outlined.exit.exitStub
!LLVMIR:  omp_loop.body:                                    ; preds = %omp_loop.cond
!LLVMIR:    br label %omp.wsloop.region
!LLVMIR:  omp.wsloop.region:                                ; preds = %omp_loop.body
!LLVMIR:    br label %omp.wsloop.region2
!LLVMIR:  omp.wsloop.region2:                               ; preds = %omp.wsloop.region3, %omp.wsloop.region
!LLVMIR:    %9 = phi i64 [ %19, %omp.wsloop.region3 ], [ 1, %omp.wsloop.region ]
!LLVMIR:    %10 = phi i64 [ %20, %omp.wsloop.region3 ], [ 9, %omp.wsloop.region ]
!LLVMIR:    br i1 %11, label %omp.wsloop.region3, label %omp.wsloop.region4
!LLVMIR:  omp.wsloop.region4:                               ; preds = %omp.wsloop.region2
!LLVMIR:    br label %omp.wsloop.exit
!LLVMIR:  omp.wsloop.exit:                                  ; preds = %omp.wsloop.region4
!LLVMIR:    br label %omp_loop.inc
!LLVMIR:   %omp_loop.next = add nuw i32 %omp_loop.iv, 1
!LLVMIR:    br label %omp_loop.header
!LLVMIR:  omp.wsloop.region3:                               ; preds = %omp.wsloop.region2
!LLVMIR:    @_FortranAioBeginExternalListOutput
!LLVMIR:    @_FortranAioOutputInteger64
!LLVMIR:    @_FortranAioEndIoStatement
!LLVMIR:    br label %omp.wsloop.region2
!LLVMIR:  }
end do
!FIRDialect:       omp.yield
!FIRDialect:         }) {inclusive, nowait, operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0, 0, 0]> : vector<9xi32>, schedule_val = "Static"} : (i32, i32, i32) -> ()
!FIRDialect:        omp.terminator
!FIRDialect:  }

!LLVMIRDialect:    omp.yield
!LLVMIRDialect:         }) {inclusive, nowait, operand_segment_sizes = dense<[1, 1, 1, 0, 0, 0, 0, 0, 0]> : vector<9xi32>, schedule_val = "Static"} : (i32, i32, i32) -> ()
!LLVMIRDialect:    omp.terminator
!LLVMIRDialect:  }
!LLVMIRDialect:  llvm.return
!LLVMIRDialect: }
!$OMP END DO NOWAIT
!$OMP END PARALLEL
end
