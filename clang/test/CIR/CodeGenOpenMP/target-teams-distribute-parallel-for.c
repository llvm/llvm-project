// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Host-side lowering to LLVM IR: the Tripcount host_eval and
// __tgt_target_kernel call are identical on both paths, so LLVM-HOST
// checks both; the outlined helper functions differ, checked below under
// LLVM-HOST-CIR/LLVM-HOST-OGCG.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-OGCG

// Device-side lowering to LLVM IR: compiled in two steps to mirror the
// offloading driver.
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

void explicit_target_teams_distribute_parallel_for() {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@explicit_target_teams_distribute_parallel_for
// CIR-HOST: %[[LB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[UB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%[[LB]] -> %[[ALB:.*]], %[[UB]] -> %[[AUB:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-HOST: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: } {omp.composite}
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@explicit_target_teams_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i{{.*}}>
// CIR-DEVICE: omp.parallel private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-DEVICE: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

void target_teams_distribute_parallel_for() {
#pragma omp target teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-HOST: %[[LB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[UB:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %{{.*}} : !s32i -> i32
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%[[LB]] -> %[[ALB:.*]], %[[UB]] -> %[[AUB:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest (%[[IV:.*]]) : i32 = (%[[ALB]]) to (%[[AUB]]) step (%[[ASTEP]]) {
// CIR-HOST: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: } {omp.composite}
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i{{.*}}>
// CIR-DEVICE: omp.parallel private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-DEVICE: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

void target_teams_split_distribute_parallel_for() {
#pragma omp target teams
#pragma omp distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_split_distribute_parallel_for
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest
// CIR-HOST: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: } {omp.composite}
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_split_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i{{.*}}>
// CIR-DEVICE: omp.parallel private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest
// CIR-DEVICE: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

void target_split_teams_distribute_parallel_for() {
#pragma omp target
#pragma omp teams distribute parallel for
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_split_teams_distribute_parallel_for
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST: omp.teams {
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST: omp.parallel private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST: omp.distribute {
// CIR-HOST-NEXT: omp.wsloop {
// CIR-HOST-NEXT: omp.loop_nest
// CIR-HOST: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST: cir.call @{{.*}}during
// CIR-HOST: omp.yield
// CIR-HOST: }
// CIR-HOST: } {omp.composite}
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.composite}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}
// CIR-HOST: omp.terminator
// CIR-HOST: } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_split_teams_distribute_parallel_for
// CIR-DEVICE: omp.target kernel_type(spmd) {
// CIR-DEVICE: omp.teams {
// CIR-DEVICE: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i{{.*}}>
// CIR-DEVICE: omp.parallel private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE: omp.distribute {
// CIR-DEVICE-NEXT: omp.wsloop {
// CIR-DEVICE-NEXT: omp.loop_nest
// CIR-DEVICE: cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE: cir.call @{{.*}}during
// CIR-DEVICE: omp.yield
// CIR-DEVICE: }
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.composite}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}
// CIR-DEVICE: omp.terminator
// CIR-DEVICE: } {omp.combined}

// LLVM-HOST-LABEL: define {{.*}} void @explicit_target_teams_distribute_parallel_for(
// LLVM-HOST:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM-HOST:         call i32 @__tgt_target_kernel(

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_distribute_parallel_for(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_split_distribute_parallel_for(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_split_teams_distribute_parallel_for(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         store i32 0, ptr %.omp.comb.lb, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 9, ptr %.omp.comb.ub, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 1, ptr %.omp.stride, align 4
// LLVM-HOST-OGCG-NEXT:    store i32 0, ptr %.omp.is_last, align 4
// LLVM-HOST-OGCG:         %[[TID3:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 92,
// LLVM-HOST-OGCG-SAME:      ptr %.omp.is_last, ptr %.omp.comb.lb, ptr %.omp.comb.ub, ptr %.omp.stride,
// LLVM-HOST-OGCG-SAME:      i32 1, i32 1)
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined,
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID4:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 34,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_distribute_parallel_for(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID2:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID2]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID2]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// `target_teams_distribute_parallel_for` lowers to the exact same shape as
// `explicit_target_teams_distribute_parallel_for` above on the OGCG path
// too.
// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID5:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID5]], i32 92,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined,
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID5]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID6:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID6]], i32 34,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID6]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_split_distribute_parallel_for(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID7:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID7]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID7]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID8:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID8]], i32 92,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined,
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID8]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID9:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID9]], i32 34,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID9]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_split_teams_distribute_parallel_for(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID10:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_dist_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID10]], i32 34,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.distupperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID10]])
// LLVM-HOST-CIR:         call void @__kmpc_barrier(
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID11:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID11]], i32 92,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 2, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined,
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID11]])
// LLVM-HOST-OGCG-NEXT:    ret void

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}.omp_outlined.omp_outlined(
// LLVM-HOST-OGCG:         %[[TID12:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG-SAME:      ptr {{.*}}, i32 %[[TID12]], i32 34,
// LLVM-HOST-OGCG:       omp.inner.for.body:
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:       omp.loop.exit:
// LLVM-HOST-OGCG-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID12]])
// LLVM-HOST-OGCG-NEXT:    ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// user_code.entry/worker.exit split are identical on both paths.
//
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:

// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null,

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 91,
// LLVM-DEVICE-OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined, ptr null,
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 33,
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null,

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 91,
// LLVM-DEVICE-OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined, ptr null,
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 33,
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null,

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 91,
// LLVM-DEVICE-OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined, ptr null,
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 33,
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}, ptr null,

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_for_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}..omp_par
// LLVM-DEVICE-CIR:         call void @__kmpc_barrier(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 91,
// LLVM-DEVICE-OGCG:         call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined, ptr null,
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_parallel_for_l{{.*}}_omp_outlined_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_init_4(ptr {{.*}}, i32 {{.*}}, i32 33,
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_for_static_fini(
