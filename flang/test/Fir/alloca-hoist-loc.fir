// Test that the descriptor copy made for a box load, which is hoisted into the
// alloca block of the enclosing OpenMP region, gets the location of the region
// and not the location of the load.
// RUN: fir-opt --cfg-conversion --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" --mlir-print-debuginfo %s | FileCheck %s --check-prefixes=CHECK,X86
// RUN: fir-opt --cfg-conversion --fir-to-llvm-ir="target=amdgcn-amd-amdhsa, datalayout=e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9" --mlir-print-debuginfo %s | FileCheck %s --check-prefixes=CHECK,AMDGPU

func.func private @takes_box(!fir.box<!fir.array<10xf32>>) -> ()

func.func @test_load_box_in_task(%addr : !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  omp.task {
    %c0 = arith.constant 0 : index loc(#loc_task_loop)
    %c1 = arith.constant 1 : index loc(#loc_task_loop)
    %c10 = arith.constant 10 : index loc(#loc_task_loop)
    fir.do_loop %i = %c0 to %c10 step %c1 {
      %0 = fir.load %addr : !fir.ref<!fir.box<!fir.array<10xf32>>> loc(#loc_task_load)
      fir.call @takes_box(%0) : (!fir.box<!fir.array<10xf32>>) -> () loc(#loc_task_load)
    } loc(#loc_task_loop)
    omp.terminator loc(#loc_task)
  } loc(#loc_task)
  return
}

func.func @test_load_box_in_parallel(%addr : !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  omp.parallel {
    %c0 = arith.constant 0 : index loc(#loc_par_loop)
    %c1 = arith.constant 1 : index loc(#loc_par_loop)
    %c10 = arith.constant 10 : index loc(#loc_par_loop)
    fir.do_loop %i = %c0 to %c10 step %c1 {
      %0 = fir.load %addr : !fir.ref<!fir.box<!fir.array<10xf32>>> loc(#loc_par_load)
      fir.call @takes_box(%0) : (!fir.box<!fir.array<10xf32>>) -> () loc(#loc_par_load)
    } loc(#loc_par_loop)
    omp.terminator loc(#loc_par)
  } loc(#loc_par)
  return
}

#loc_task = loc("test.f90":6:7)
#loc_task_loop = loc("test.f90":7:9)
#loc_task_load = loc("test.f90":8:11)
#loc_par = loc("test.f90":10:7)
#loc_par_loop = loc("test.f90":11:9)
#loc_par_load = loc("test.f90":12:11)

// CHECK-LABEL: llvm.func @test_load_box_in_task(
// CHECK:         omp.task {
// CHECK-NEXT:      %[[TASK_ONE:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_TASK:loc[0-9]*]])
// X86-NEXT:        %[[TASK_COPY:.*]] = llvm.alloca %[[TASK_ONE]] x !llvm.struct<{{.*}}> {{.*}} : (i32) -> !llvm.ptr loc(#[[LOC_TASK]])
// AMDGPU-NEXT:     %[[TASK_ALLOCA:.*]] = llvm.alloca %[[TASK_ONE]] x !llvm.struct<{{.*}}> {{.*}} : (i32) -> !llvm.ptr<5> loc(#[[LOC_TASK]])
// AMDGPU-NEXT:     %[[TASK_COPY:.*]] = llvm.addrspacecast %[[TASK_ALLOCA]] : !llvm.ptr<5> to !llvm.ptr loc(#[[LOC_TASK]])
// CHECK:           "llvm.intr.memcpy"(%[[TASK_COPY]], {{.*}}) {{.*}} loc(#[[LOC_TASK_LOAD:loc[0-9]*]])
// CHECK:           llvm.call @takes_box(%[[TASK_COPY]]) {{.*}} loc(#[[LOC_TASK_LOAD]])

// CHECK-LABEL: llvm.func @test_load_box_in_parallel(
// CHECK:         omp.parallel {
// CHECK-NEXT:      %[[PAR_ONE:.*]] = llvm.mlir.constant(1 : i32) : i32 loc(#[[LOC_PAR:loc[0-9]*]])
// X86-NEXT:        %[[PAR_COPY:.*]] = llvm.alloca %[[PAR_ONE]] x !llvm.struct<{{.*}}> {{.*}} : (i32) -> !llvm.ptr loc(#[[LOC_PAR]])
// AMDGPU-NEXT:     %[[PAR_ALLOCA:.*]] = llvm.alloca %[[PAR_ONE]] x !llvm.struct<{{.*}}> {{.*}} : (i32) -> !llvm.ptr<5> loc(#[[LOC_PAR]])
// AMDGPU-NEXT:     %[[PAR_COPY:.*]] = llvm.addrspacecast %[[PAR_ALLOCA]] : !llvm.ptr<5> to !llvm.ptr loc(#[[LOC_PAR]])
// CHECK:           "llvm.intr.memcpy"(%[[PAR_COPY]], {{.*}}) {{.*}} loc(#[[LOC_PAR_LOAD:loc[0-9]*]])
// CHECK:           llvm.call @takes_box(%[[PAR_COPY]]) {{.*}} loc(#[[LOC_PAR_LOAD]])

// CHECK-DAG: #[[LOC_TASK]] = loc("test.f90":6:7)
// CHECK-DAG: #[[LOC_TASK_LOAD]] = loc("test.f90":8:11)
// CHECK-DAG: #[[LOC_PAR]] = loc("test.f90":10:7)
// CHECK-DAG: #[[LOC_PAR_LOAD]] = loc("test.f90":12:11)
