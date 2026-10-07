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
// offloading driver. The kernel entry sequence is the same on both paths.
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

// This is a target SPMD construct, so the host also computes the loop's
// trip count and forwards it via host_eval block arguments.
void target_parallel_for() {
#pragma omp target
#pragma omp parallel
#pragma omp for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_parallel_for

// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The trip count is computed before the omp.target, from the original
// bounds/step: tripCount = (10 - (0 - 1 + 1)) / 1 = 10. Then 0, the trip
// count, and 1 are each cast to a builtin integer and forwarded as the
// omp.target's host_eval operands.
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST-NEXT: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST-NEXT: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST-NEXT: omp.target kernel_type(spmd) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT: omp.parallel {

// CIR-HOST: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

// The loop inside the target region uses the host_eval block arguments
// instead of recomputing the bounds.
// CIR-HOST-NEXT: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
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
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.parallel {

// The two allocas and their address-space casts can be emitted in either
// relative order, so match them unordered (CHECK-DAG) rather than assuming
// a specific interleaving.
// CIR-DEVICE-DAG: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-DAG: %[[IV_CAST:.*]] = cir.cast address_space %[[IV_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
// CIR-DEVICE-DAG: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i, target_address_space(5)>
// CIR-DEVICE-DAG: %[[I_CAST:.*]] = cir.cast address_space %[[I_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>

// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// The device has no host_eval operands, so it recomputes the same trip
// count inside the region.
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32

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
// inside omp.target. This is a target SPMD construct, so the omp.target is
// marked kernel_type(spmd) with host_eval'd loop bounds, just like the
// explicitly nested directives above.
void combined_target_parallel_for() {
#pragma omp target parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@combined_target_parallel_for
// The loop only runs if its precondition `0 < 10` holds.
// CIR-HOST: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-HOST-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-HOST-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-HOST-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-HOST-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-HOST-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-HOST-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-HOST: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %{{.*}} -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST: omp.parallel {
// CIR-HOST: %[[CI_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.wsloop private(@{{.*}} %[[CI_ALLOCA]] -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT: omp.loop_nest (%[[CIV:.*]]) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: }
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@combined_target_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.parallel {
// The loop only runs if its precondition `0 < 10` holds.
// CIR-DEVICE: %[[PRE_L:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[PRE_R:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[PRECOND:.*]] = cir.cmp lt %[[PRE_L]], %[[PRE_R]] : !s32i
// tripCount = (10 - (0 - 1 + 1)) / 1 = 10
// CIR-DEVICE-NEXT: %[[HI:.*]] = cir.const #cir.int<10> : !s32i
// CIR-DEVICE-NEXT: %[[LO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[STEP_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[LO_MINUS_STEP:.*]] = cir.sub nsw %[[LO]], %[[STEP_A]] : !s32i
// CIR-DEVICE-NEXT: %[[ONE_A:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[ADJ_LO:.*]] = cir.add nsw %[[LO_MINUS_STEP]], %[[ONE_A]] : !s32i
// CIR-DEVICE-NEXT: %[[SPAN:.*]] = cir.sub nsw %[[HI]], %[[ADJ_LO]] : !s32i
// CIR-DEVICE-NEXT: %[[STEP_B:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_CIR:.*]] = cir.div %[[SPAN]], %[[STEP_B]] : !s32i
// CIR-DEVICE-NEXT: %[[TC_ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE-NEXT: %[[TRIPCOUNT_SEL:.*]] = cir.select if %[[PRECOND]] then %[[TRIPCOUNT_CIR]] else %[[TC_ZERO]] : (!cir.bool, !s32i, !s32i) -> !s32i
// CIR-DEVICE: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !s32i -> i32
// CIR-DEVICE: omp.wsloop private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%[[TRIPCOUNT]]) step (%{{.*}}) {
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: }
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

// Host wrappers: the __tgt_target_kernel call, the offload-failed fallback
// branch, and the call into the offloading-entry function are identical on
// both paths. This is a target SPMD construct, so CIR's kernel-launch
// arguments carry the host_eval'd trip count (10) in the trip count field
// (index 8); OGCG doesn't forward a host-computed trip count here, so it
// leaves that field 0.
//
// LLVM-HOST-LABEL: define {{.*}} void @target_parallel_for(
// LLVM-HOST:         %[[KARGS:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS]], i32 0, i32 8
// LLVM-HOST-CIR-NEXT:  store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM-HOST-OGCG-NEXT: store i64 0, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:         br i1 %{{.*}}, label %[[FAILED:[^,]+]], label %{{.*}}
// LLVM-HOST:       {{^}}[[FAILED]]:
// LLVM-HOST-NEXT:    call void @__omp_offloading_{{.*}}_target_parallel_for_l{{[0-9]+}}(

// CIR groups all its helper functions together up front, since the
// MLIR-to-LLVM translation emits them during the lowering pass itself;
// OGCG's direct AST walk emits each one in source order instead,
// interleaved with other constructs (same as target-parallel.c). The
// checks are the same either way, just placed where each path actually
// emits them (compare the LLVM-HOST-OGCG copy of this block further down).
// LLVM-HOST-CIR-LABEL: define {{.*}} void @combined_target_parallel_for(
// LLVM-HOST-CIR:         %[[KARGS2:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS2]], i32 0, i32 8
// LLVM-HOST-CIR-NEXT:    store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-CIR:       {{^}}[[FAILED2]]:
// LLVM-HOST-CIR-NEXT:    call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{[0-9]+}}(

// The outer wrapper forks the parallel region via __kmpc_fork_call.
//
// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_for_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[OUTLINED:[^)]+]])

// The worksharing-loop body: CIR and OGCG lower the loop through different
// runtime APIs here, but both call during() once per iteration and end
// with a barrier.
// LLVM-HOST-CIR:       define internal void @[[OUTLINED]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE:.*]], align 4
// LLVM-HOST-CIR:         %[[TID:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %{{.*}}, ptr %[[LB]], ptr %[[UB]], ptr %[[STRIDE]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP:.*]] = icmp ult i32 %[[IV]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP]], label %[[LOOP_BODY:[^,]+]], label %[[LOOP_EXIT:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT]] = add nuw i32 %[[IV]], 1

// LLVM-HOST-OGCG:       define internal void @[[OUTLINED]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID2:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST]], ptr %[[OMP_LB]], ptr %[[OMP_UB]], ptr %[[OMP_STRIDE]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         %[[UB_VAL:.*]] = load i32, ptr %[[OMP_UB]], align 4
// LLVM-HOST-OGCG-NEXT:    %[[CMP:.*]] = icmp sgt i32 %[[UB_VAL]], 9
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP]], label %[[COND_TRUE:[^,]+]], label %[[COND_FALSE:[^ ,]+]]
// LLVM-HOST-OGCG:         %[[COND:.*]] = phi i32 [ 9, %[[COND_TRUE]] ], [ %{{.*}}, %[[COND_FALSE]] ]
// LLVM-HOST-OGCG-NEXT:    store i32 %[[COND]], ptr %[[OMP_UB]], align 4
// LLVM-HOST-OGCG:         %[[CMP1:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP1]], label %[[FOR_BODY:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FOR_BODY]]:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_barrier(ptr {{.*}}, i32 %[[TID2]])

// OGCG's copy of combined_target_parallel_for's host wrapper: identical
// content to the LLVM-HOST-CIR block above, just emitted after
// target_parallel_for's entry+outlined functions instead of before them.
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @combined_target_parallel_for(
// LLVM-HOST-OGCG:         %[[KARGS2:.*]] = alloca %struct.__tgt_kernel_arguments, align 8
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %[[KARGS2]], i32 0, i32 8
// LLVM-HOST-OGCG-NEXT:    store i64 0, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FAILED2]]:
// LLVM-HOST-OGCG-NEXT:    call void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{[0-9]+}}(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @[[OUTLINED2:[^)]+]])

// LLVM-HOST-CIR:       define internal void @[[OUTLINED2]](
// LLVM-HOST-CIR:         store i32 0, ptr %[[LB2:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 9, ptr %[[UB2:.*]], align 4
// LLVM-HOST-CIR-NEXT:    store i32 1, ptr %[[STRIDE2:.*]], align 4
// LLVM-HOST-CIR:         %[[TID3:.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %{{.*}}, ptr %[[LB2]], ptr %[[UB2]], ptr %[[STRIDE2]],
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %[[IV2:.*]] = phi i32 [ 0, %{{.*}} ], [ %[[NEXT2:[^,]+]], %{{.*}} ]
// LLVM-HOST-CIR:         %[[LOOP_CMP2:.*]] = icmp ult i32 %[[IV2]], %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %[[LOOP_CMP2]], label %[[LOOP_BODY2:[^,]+]], label %[[LOOP_EXIT2:[^ ,]+]]
// LLVM-HOST-CIR:       {{^}}[[LOOP_EXIT2]]:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(ptr {{.*}}, i32 %{{.*}})
// LLVM-HOST-CIR:       {{^}}[[LOOP_BODY2]]:
// LLVM-HOST-CIR:         call void @during(i32
// LLVM-HOST-CIR:         %[[NEXT2]] = add nuw i32 %[[IV2]], 1

// Unlike the non-combined form above, OGCG does not emit a trailing
// barrier after this combined construct's worksharing loop: since the
// loop is the entire body of the parallel region, the implicit join at
// the end of the parallel region already synchronizes the threads. CIR
// doesn't yet apply that optimization, so it emits one either way.
// LLVM-HOST-OGCG:       define internal void @[[OUTLINED2]](
// LLVM-HOST-OGCG:         store i32 0, ptr %[[OMP_LB2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %[[OMP_UB2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %[[OMP_STRIDE2:.*]], align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %[[OMP_IS_LAST2:.*]], align 4
// LLVM-HOST-OGCG:         %[[TID4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 34,
// LLVM-HOST-OGCG-SAME:      ptr %[[OMP_IS_LAST2]], ptr %[[OMP_LB2]], ptr %[[OMP_UB2]], ptr %[[OMP_STRIDE2]],
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:         %[[UB_VAL2:.*]] = load i32, ptr %[[OMP_UB2]], align 4
// LLVM-HOST-OGCG-NEXT:    %[[CMP2:.*]] = icmp sgt i32 %[[UB_VAL2]], 9
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP2]], label %[[COND_TRUE2:[^,]+]], label %[[COND_FALSE2:[^ ,]+]]
// LLVM-HOST-OGCG:         %[[COND2:.*]] = phi i32 [ 9, %[[COND_TRUE2]] ], [ %{{.*}}, %[[COND_FALSE2]] ]
// LLVM-HOST-OGCG-NEXT:    store i32 %[[COND2]], ptr %[[OMP_UB2]], align 4
// LLVM-HOST-OGCG:         %[[CMP3:.*]] = icmp sle i32 %{{.*}}, %{{.*}}
// LLVM-HOST-OGCG-NEXT:    br i1 %[[CMP3]], label %[[FOR_BODY2:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FOR_BODY2]]:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-HOST-OGCG-NEXT:    ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// branch into the user code are identical on both paths.
//
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_for_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE]]:

// Unlike target-parallel.c's generic kernel, this is an SPMD kernel, so
// neither path needs a wrapper here: CIR also passes null, since every
// thread already runs the parallel region directly.
// LLVM-DEVICE:       call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @[[DEV_OUTLINED:[^,]+]], ptr null, ptr {{.*}}, i64 0, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()

// Past the kernel entry, CIR and OGCG use completely different device
// runtime APIs for the loop (a newer driver+callback API for CIR vs one
// big function for OGCG), so there isn't much to share beyond the kernel
// entry checked above.
//
// LLVM-DEVICE-CIR:       define internal void @[[DEV_OUTLINED]](
// LLVM-DEVICE-CIR:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @[[DEV_LOOP_BODY:[^,]+]],
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-CIR:       define internal void @[[DEV_LOOP_BODY]](i32
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_OUTLINED]](
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_combined_target_parallel_for_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE2:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE2]]:

// LLVM-DEVICE:       call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @[[DEV_OUTLINED2:[^,]+]], ptr null, ptr {{.*}}, i64 0, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()

// LLVM-DEVICE-CIR:       define internal void @[[DEV_OUTLINED2]](
// LLVM-DEVICE-CIR:         call void @__kmpc_for_static_loop_4u(ptr {{.*}}, ptr @[[DEV_LOOP_BODY2:[^,]+]],
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-CIR:       define internal void @[[DEV_LOOP_BODY2]](i32
// LLVM-DEVICE-CIR:         call void @during(i32

// LLVM-DEVICE-OGCG:       define internal void @[[DEV_OUTLINED2]](
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(
