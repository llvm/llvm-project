! TODO Convert this file into a bunch of lit tests for each conversion step.

! RUN: bbc -fopenmp -emit-fir --openmp-enable-delayed-privatization -hlfir=false %s -o - 

subroutine delayed_privatization()
  implicit none
  integer :: var1
  integer :: var2

  var1 = 111
  var2 = 222

!$OMP PARALLEL FIRSTPRIVATE(var1, var2)
  var1 = var1 + var2 + 2
!$OMP END PARALLEL

end subroutine

! -----------------------------------------
! ## This is what flang emits with the PoC:
! -----------------------------------------
!
! ----------------------------
! ### Conversion to FIR + OMP:
! ----------------------------
!module {
!  func.func @_QPdelayed_privatization() {
!    %0 = fir.alloca i32 {bindc_name = "var1", uniq_name = "_QFdelayed_privatizationEvar1"}
!    %1 = fir.alloca i32 {bindc_name = "var2", uniq_name = "_QFdelayed_privatizationEvar2"}
!    %c111_i32 = arith.constant 111 : i32
!    fir.store %c111_i32 to %0 : !fir.ref<i32>
!    %c222_i32 = arith.constant 222 : i32
!    fir.store %c222_i32 to %1 : !fir.ref<i32>
!    omp.parallel private(@var1.privatizer %0, @var2.privatizer %1 : !fir.ref<i32>, !fir.ref<i32>) {
!    ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
!      %2 = fir.load %arg0 : !fir.ref<i32>
!      %3 = fir.load %arg1 : !fir.ref<i32>
!      %4 = arith.addi %2, %3 : i32
!      %c2_i32 = arith.constant 2 : i32
!      %5 = arith.addi %4, %c2_i32 : i32
!      fir.store %5 to %arg0 : !fir.ref<i32>
!      omp.terminator
!    }
!    return
!  }
!  "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var1.privatizer"}> ({
!  ^bb0(%arg0: !fir.ref<i32>):
!    %0 = fir.alloca i32 {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatizationEvar1"}
!    %1 = fir.load %arg0 : !fir.ref<i32>
!    fir.store %1 to %0 : !fir.ref<i32>
!    omp.yield(%0 : !fir.ref<i32>)
!  }) : () -> ()
!  "omp.private"() <{function_type = (!fir.ref<i32>) -> !fir.ref<i32>, sym_name = "var2.privatizer"}> ({
!  ^bb0(%arg0: !fir.ref<i32>):
!    %0 = fir.alloca i32 {bindc_name = "var2", pinned, uniq_name = "_QFdelayed_privatizationEvar2"}
!    %1 = fir.load %arg0 : !fir.ref<i32>
!    fir.store %1 to %0 : !fir.ref<i32>
!    omp.yield(%0 : !fir.ref<i32>)
!  }) : () -> ()
!
! -----------------------------
! ### Conversion to LLVM + OMP:
! -----------------------------
!module {
!  llvm.func @_QPdelayed_privatization() {
!    %0 = llvm.mlir.constant(1 : i64) : i64
!    %1 = llvm.alloca %0 x i32 {bindc_name = "var1"} : (i64) -> !llvm.ptr
!    %2 = llvm.mlir.constant(1 : i64) : i64
!    %3 = llvm.alloca %2 x i32 {bindc_name = "var2"} : (i64) -> !llvm.ptr
!    %4 = llvm.mlir.constant(111 : i32) : i32
!    llvm.store %4, %1 : i32, !llvm.ptr
!    %5 = llvm.mlir.constant(222 : i32) : i32
!    llvm.store %5, %3 : i32, !llvm.ptr
!    omp.parallel private(@var1.privatizer %1, @var2.privatizer %3 : !llvm.ptr, !llvm.ptr) {
!    ^bb0(%arg0: !llvm.ptr, %arg1: !llvm.ptr):
!      %6 = llvm.load %arg0 : !llvm.ptr -> i32
!      %7 = llvm.load %arg1 : !llvm.ptr -> i32
!      %8 = llvm.add %6, %7  : i32
!      %9 = llvm.mlir.constant(2 : i32) : i32
!      %10 = llvm.add %8, %9  : i32
!      llvm.store %10, %arg0 : i32, !llvm.ptr
!      omp.terminator
!    }
!    llvm.return
!  }
!  "omp.private"() <{function_type = (!llvm.ptr) -> !llvm.ptr, sym_name = "var1.privatizer"}> ({
!  ^bb0(%arg0: !llvm.ptr):
!    %0 = llvm.mlir.constant(1 : i64) : i64
!    %1 = llvm.alloca %0 x i32 {bindc_name = "var1", pinned} : (i64) -> !llvm.ptr
!    %2 = llvm.load %arg0 : !llvm.ptr -> i32
!    llvm.store %2, %1 : i32, !llvm.ptr
!    omp.yield(%1 : !llvm.ptr)
!  }) : () -> ()
!  "omp.private"() <{function_type = (!llvm.ptr) -> !llvm.ptr, sym_name = "var2.privatizer"}> ({
!  ^bb0(%arg0: !llvm.ptr):
!    %0 = llvm.mlir.constant(1 : i64) : i64
!    %1 = llvm.alloca %0 x i32 {bindc_name = "var2", pinned} : (i64) -> !llvm.ptr
!    %2 = llvm.load %arg0 : !llvm.ptr -> i32
!    llvm.store %2, %1 : i32, !llvm.ptr
!    omp.yield(%1 : !llvm.ptr)
!  }) : () -> ()
!}
!
! --------------------------
! ### Conversion to LLVM IR:
! --------------------------
!%struct.ident_t = type { i32, i32, i32, i32, ptr }

!@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
!@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8

!define void @_QPdelayed_privatization() {
!  %structArg = alloca { ptr, ptr }, align 8
!  %1 = alloca i32, i64 1, align 4
!  %2 = alloca i32, i64 1, align 4
!  store i32 111, ptr %1, align 4
!  store i32 222, ptr %2, align 4
!  br label %entry

!entry:                                            ; preds = %0
!  %omp_global_thread_num = call i32 @__kmpc_global_thread_num(ptr @1)
!  br label %omp_parallel

!omp_parallel:                                     ; preds = %entry
!  %gep_ = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 0
!  store ptr %1, ptr %gep_, align 8
!  %gep_2 = getelementptr { ptr, ptr }, ptr %structArg, i32 0, i32 1
!  store ptr %2, ptr %gep_2, align 8
!  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @_QPdelayed_privatization..omp_par, ptr %structArg)
!  br label %omp.par.outlined.exit

!omp.par.outlined.exit:                            ; preds = %omp_parallel
!  br label %omp.par.exit.split

!omp.par.exit.split:                               ; preds = %omp.par.outlined.exit
!  ret void
!}

!; Function Attrs: nounwind
!define internal void @_QPdelayed_privatization..omp_par(ptr noalias %tid.addr, ptr noalias %zero.addr, ptr %0) #0 {
!omp.par.entry:
!  %gep_ = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 0
!  %loadgep_ = load ptr, ptr %gep_, align 8
!  %gep_1 = getelementptr { ptr, ptr }, ptr %0, i32 0, i32 1
!  %loadgep_2 = load ptr, ptr %gep_1, align 8
!  %tid.addr.local = alloca i32, align 4
!  %1 = load i32, ptr %tid.addr, align 4
!  store i32 %1, ptr %tid.addr.local, align 4
!  %tid = load i32, ptr %tid.addr.local, align 4
!  %2 = alloca i32, i64 1, align 4
!  %3 = load i32, ptr %loadgep_, align 4
!  store i32 %3, ptr %2, align 4
!  %4 = alloca i32, i64 1, align 4
!  %5 = load i32, ptr %loadgep_2, align 4
!  store i32 %5, ptr %4, align 4
!  br label %omp.par.region

!omp.par.region:                                   ; preds = %omp.par.entry
!  br label %omp.par.region1

!omp.par.region1:                                  ; preds = %omp.par.region
!  %6 = load i32, ptr %2, align 4
!  %7 = load i32, ptr %4, align 4
!  %8 = add i32 %6, %7
!  %9 = add i32 %8, 2
!  store i32 %9, ptr %2, align 4
!  br label %omp.region.cont

!omp.region.cont:                                  ; preds = %omp.par.region1
!  br label %omp.par.pre_finalize

!omp.par.pre_finalize:                             ; preds = %omp.region.cont
!  br label %omp.par.outlined.exit.exitStub

!omp.par.outlined.exit.exitStub:                   ; preds = %omp.par.pre_finalize
!  ret void
!}

!; Function Attrs: nounwind
!declare i32 @__kmpc_global_thread_num(ptr) #0

!; Function Attrs: nounwind
!declare !callback !2 void @__kmpc_fork_call(ptr, i32, ptr, ...) #0
