! This test checks that chunk size is passed correctly when lowering of
! OpenMP DO Directive(Worksharing)

! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   tco | FileCheck %s --check-prefix=LLVMIR

program wsloop
        integer :: i
        integer :: chunk
!FIRDialect: func @_QQmain()
!LLVMIRDialect: func @_QQmain()

!LLVMIR: define void @_QQmain()
!LLVMIR:call i32 @__kmpc_global_thread_num{{.*}}
!LLVMIR:  br label %omp_parallel

!$OMP PARALLEL

!LLVMIR: omp_parallel:                                     ; preds = %0
!LLVMIR:   @__kmpc_fork_call
!$OMP DO SCHEDULE(static, 4)

!LLVMIR:  define internal void @_QQmain..omp_par
!LLVMIR:   %p.lastiter{{.*}} = alloca i32, align 4
!LLVMIR:   %p.lowerbound{{.*}} = alloca i32, align 4
!LLVMIR:   %p.upperbound{{.*}} = alloca i32, align 4
!LLVMIR:   %p.stride{{.*}} = alloca i32, align 4
!LLVMIR:  omp_loop.preheader:                               ; preds = %omp.par.region1
!LLVMIR:    @__kmpc_global_thread_num
!LLVMIR:    call void @__kmpc_for_static_init_4u(%{{.*}}, i32 %{{.*}}, i32 34, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 4)
!LLVMIR:    br label %omp_loop.header
!LLVMIR:  omp_loop.header:                                  ; preds = %omp_loop.inc, %omp_loop.preheader
!LLVMIR:    %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]

do i=1, 9
  print*, i


!LLVMIR:   br label %omp_loop.cond
!LLVMIR: omp_loop.cond:                                    ; preds = %omp_loop.header
!LLVMIR:   %omp_loop.cmp = icmp ult i32 %{{.*}}, %{{.*}}
!LLVMIR:   br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
!LLVMIR: omp_loop.exit:                                    ; preds = %omp_loop.cond
!LLVMIR:   call void @__kmpc_for_static_fini(%struct.ident_t* @{{.*}}, i32 %omp_global_thread_num2)

end do
!$OMP END DO NOWAIT
!$OMP DO SCHEDULE(static, 2+2)

!LLVMIR:  omp_loop.preheader{{.*}}:                               ; preds = %omp_loop.after
!LLVMIR:    @__kmpc_global_thread_num
!LLVMIR:    call void @__kmpc_for_static_init_4u(%{{.*}}, i32 %{{.*}}, i32 34, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 4)
!LLVMIR:    br label %omp_loop.header{{.*}}
!LLVMIR:  omp_loop.header{{.*}}:                                  ; preds = %omp_loop.inc{{.*}}, %omp_loop.preheader{{.*}}
!LLVMIR:    %omp_loop.iv{{.*}} = phi i32 [ 0, %omp_loop.preheader{{.*}} ], [ %omp_loop.next{{.*}}, %omp_loop.inc{{.*}} ]

do i=1, 9
  print*, i*2

!LLVMIR:   br label %omp_loop.cond{{.*}}
!LLVMIR: omp_loop.cond{{.*}}:                                    ; preds = %omp_loop.header{{.*}}
!LLVMIR:   %omp_loop.cmp{{.*}} = icmp ult i32 %{{.*}}, %{{.*}}
!LLVMIR:   br i1 %omp_loop.cmp{{.*}}, label %omp_loop.body{{.*}}, label %omp_loop.exit{{.*}}
!LLVMIR: omp_loop.exit{{.*}}:                                    ; preds = %omp_loop.cond{{.*}}
!LLVMIR:   call void @__kmpc_for_static_fini(%struct.ident_t* @{{.*}}, i32 %omp_global_thread_num{{.*}})

end do
!$OMP END DO NOWAIT
chunk = 6
!$OMP DO SCHEDULE(static, chunk)
!LLVMIR: omp_loop.after{{.*}}:                                  ; preds = %omp_loop.exit{{.*}}
!LLVMIR:   store i32 6, i32* %[[CHUNK_ADDR:.*]], align 4
!LLVMIR:   %[[CHUNK_VAL:.*]] = load i32, i32* %[[CHUNK_ADDR]], align 4

!LLVMIR:  omp_loop.preheader{{.*}}:                               ; preds = %omp_loop.after
!LLVMIR:    @__kmpc_global_thread_num
!LLVMIR:    call void @__kmpc_for_static_init_4u(%{{.*}}, i32 %{{.*}}, i32 34, i32* %{{.*}}, i32* %{{.*}}, i32* %{{.*}}, i32 1, i32 %[[CHUNK_VAL]])
!LLVMIR:    br label %omp_loop.header{{.*}}
!LLVMIR:  omp_loop.header{{.*}}:                                  ; preds = %omp_loop.inc{{.*}}, %omp_loop.preheader{{.*}}
!LLVMIR:    %omp_loop.iv{{.*}} = phi i32 [ 0, %omp_loop.preheader{{.*}} ], [ %omp_loop.next{{.*}}, %omp_loop.inc{{.*}} ]

do i=1, 9
   print*, i*3
end do
!$OMP END DO NOWAIT

!LLVMIR: omp_loop.body{{.*}}:                                    ; preds = %omp_loop.cond
!LLVMIR:   %{{.*}} = add i32 %{{.*}}, %{{.*}}
!LLVMIR:   %{{.*}} = mul i32 %{{.*}}, 1
!LLVMIR:   %{{.*}} = add i32 %{{.*}}, 1
!LLVMIR:   br label %omp.wsloop.region{{.*}}
!LLVMIR: omp.wsloop.region{{.*}}:                                ; preds = %omp_loop.body{{.*}}

!LLVMIR: omp_loop.body:                                    ; preds = %omp_loop.cond
!LLVMIR:   %{{.*}} = add i32 %{{.*}}, %{{.*}}
!LLVMIR:   %{{.*}} = mul i32 %{{.*}}, 1
!LLVMIR:   %{{.*}} = add i32 %{{.*}}, 1
!LLVMIR:   br label %omp.wsloop.region
!LLVMIR: omp.wsloop.region:                                ; preds = %omp_loop.body

!$OMP END PARALLEL
end
