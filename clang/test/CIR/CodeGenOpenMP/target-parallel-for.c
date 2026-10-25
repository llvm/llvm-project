// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Host-side lowering to LLVM IR: mostly identical between CIR and OGCG.
// The loop body is lowered through a different OpenMP runtime API on each
// path (see the LLVM-HOST-CIR/LLVM-HOST-OGCG split below), and the two
// paths emit functions in a different order (same as target-parallel.c).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-OGCG

// Device-side lowering to LLVM IR: compiled in two steps to mirror the
// offloading driver. The kernel entry sequence is the same on both paths,
// aside from the wrapper-vs-null difference seen in target-parallel.c.
// Past that point, CIR and OGCG lower the worksharing loop through
// different runtime APIs, so they're checked separately under
// LLVM-DEVICE-CIR/LLVM-DEVICE-OGCG.
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -fclangir -emit-llvm-bc %s -o %t-cir-host.bc
// RUN: %clang_cc1 -fopenmp -triple x86_64-linux-gnu -fopenmp-targets=amdgcn-amd-amdhsa \
// RUN:   -emit-llvm-bc %s -o %t-ogcg-host.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-cir-host.bc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-DEVICE,LLVM-DEVICE-CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-ogcg-host.bc \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-DEVICE,LLVM-DEVICE-OGCG

void during(int);

void target_parallel_for() {
#pragma omp target
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_parallel_for
// CIR-HOST: omp.target kernel_type(generic) {
// CIR-HOST: omp.parallel {

// CIR-HOST: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

// The loop bounds are normalized to a `[0, tripCount)` range; tripCount is
// computed from the original bounds/step.
// CIR-HOST: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-HOST: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

// CIR-HOST: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-HOST: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-HOST: cir.store align(4) %[[IV_CIR]], %[[IV_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: }
// CIR-HOST: omp.terminator
// CIR-HOST: omp.terminator
// CIR-HOST: }

// CIR-DEVICE: cir.func{{.*}}@target_parallel_for
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE: omp.parallel {

// The two allocas and their address-space casts can be emitted in either
// relative order, so match them unordered (CHECK-DAG) rather than assuming
// a specific interleaving.
// CIR-DEVICE-DAG: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-DAG: %[[IV_CAST:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-DAG: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-DAG: %[[I_CAST:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE: %[[C10_CIR:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE: %[[C0_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE: %[[C1_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-DEVICE: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

// CIR-DEVICE: omp.wsloop private(@{{.*}} %[[I_CAST]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ZERO]]) to (%[[TRIPCOUNT]]) step (%[[ONE]]) {
// CIR-DEVICE: %[[IV_CIR:.*]] = cir.builtin_int_cast %[[IV]] : i32 -> !s32i
// CIR-DEVICE: cir.store align(4) %[[IV_CIR]], %[[IV_CAST]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: }
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: }

// The combined `target parallel for` directive decomposes into `target`,
// `parallel` and `for` leaves and lowers to the same nesting as the explicit
// target/parallel/for above: an omp.wsloop + omp.loop_nest inside omp.parallel
// inside omp.target.
void combined_target_parallel_for() {
#pragma omp target parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@combined_target_parallel_for
// CIR-HOST: omp.target kernel_type(generic) {
// CIR-HOST: omp.parallel {
// CIR-HOST: %[[CI_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.wsloop private(@{{.*}} %[[CI_ALLOCA]] -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT: omp.loop_nest (%[[CIV:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: }
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@combined_target_parallel_for
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE: omp.parallel {
// CIR-DEVICE: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: }
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

// Host wrappers: the __tgt_target_kernel call, the omp_offload.failed
// branch, and the call into the offloading-entry function are identical on
// both paths.
//
// LLVM-HOST-LABEL: define {{.*}} void @target_parallel_for(
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:       omp_offload.failed:
// LLVM-HOST:         call void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}(

// CIR groups all its helper functions together up front, since the
// MLIR-to-LLVM translation emits them during the lowering pass itself;
// OGCG's direct AST walk emits each one in source order instead,
// interleaved with other constructs (same as target-parallel.c). The
// checks are the same either way, just placed where each path actually
// emits them (compare the LLVM-HOST-OGCG copy of this block further down).
// LLVM-HOST-CIR-LABEL: define {{.*}} void @combined_target_parallel_for(
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:       omp_offload.failed:
// LLVM-HOST-CIR:         call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}(

// The outer wrapper forks the parallel region via __kmpc_fork_call. CIR
// and OGCG just name the forked function differently, which we don't
// check since it doesn't matter.
//
// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}})

// The worksharing-loop body: CIR and OGCG lower the loop through different
// runtime APIs here, but both call during() once per iteration and end
// with a barrier.
// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par(
// LLVM-HOST-CIR:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %p.upperbound, align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM-HOST-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-HOST-OGCG:         %[[TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-HOST-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         %[[UB:.*]] = load i32, ptr %.omp.ub, align 4
// LLVM-HOST-OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB]], 9
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP]], label %cond.true, label %cond.false
// LLVM-HOST-OGCG:       cond.end:
// LLVM-HOST-OGCG:         %[[COND:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-HOST-OGCG-NEXT:    store i32 %[[COND]], ptr %.omp.ub, align 4
// LLVM-HOST-OGCG:       omp.inner.for.cond:
// LLVM-HOST-OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP1]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_barrier(ptr {{.*}}, i32 %[[TID2]])

// OGCG's copy of combined_target_parallel_for's host wrapper: identical
// content to the LLVM-HOST-CIR block above, just emitted after
// target_parallel_for's entry+outlined functions instead of before them.
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @combined_target_parallel_for(
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:       omp_offload.failed:
// LLVM-HOST-OGCG:         call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par(
// LLVM-HOST-CIR:         store i32 0, ptr %p.lowerbound, align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %p.upperbound, align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %p.stride, align 4
// LLVM-HOST-CIR:         %[[TID3:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %omp_loop.next = add nuw i32 %omp_loop.iv, 1

// Unlike the non-combined form above, OGCG does not emit a trailing
// barrier after this combined construct's worksharing loop: since the
// loop is the entire body of the parallel region, the implicit join at
// the end of the parallel region already synchronizes the threads. CIR
// doesn't yet apply that optimization, so it emits one either way.
// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         store i32 0, ptr %.omp.lb, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %.omp.ub, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-HOST-OGCG:         %[[TID4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 34,
// LLVM-HOST-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.lb, ptr %.omp.ub, ptr %.omp.stride,
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         %[[UB2:.*]] = load i32, ptr %.omp.ub, align 4
// LLVM-HOST-OGCG-NEXT:    %[[CMP2:.*]] = icmp sgt i32 %[[UB2]], 9
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP2]], label %cond.true, label %cond.false
// LLVM-HOST-OGCG:       cond.end:
// LLVM-HOST-OGCG:         %[[COND2:.*]] = phi i32 [ 9, %cond.true ], [ %{{.*}}, %cond.false ]
// LLVM-HOST-OGCG-NEXT:    store i32 %[[COND2]], ptr %.omp.ub, align 4
// LLVM-HOST-OGCG:       omp.inner.for.cond:
// LLVM-HOST-OGCG:         %[[CMP3:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP3]], label %omp.inner.for.body, label %omp.inner.for.end
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-HOST-OGCG-NEXT:    ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// user_code.entry split are identical on both paths.
//
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:

// CIR passes a wrapper function to __kmpc_parallel_60; OGCG passes null
// instead, since it doesn't need one here (same as target-parallel.c).
// LLVM-DEVICE-CIR:   call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par{{.*}}, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par{{.*}}.wrapper, ptr {{.*}}, i64 0, i32 0)
// LLVM-DEVICE-OGCG:  call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 0, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()

// Past the kernel entry, CIR and OGCG use completely different device
// runtime APIs for the loop (a newer driver+callback API for CIR vs one
// big function for OGCG), so there isn't much to share beyond the kernel
// entry checked above.
//
// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par,
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}..omp_par(i32
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_combined_target_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:

// LLVM-DEVICE-CIR:   call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par{{.*}}, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par{{.*}}.wrapper, ptr {{.*}}, i64 0, i32 0)
// LLVM-DEVICE-OGCG:  call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}_omp_outlined, ptr null, ptr {{.*}}, i64 0, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par,
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}..omp_par(i32
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(
