// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Host-side lowering to LLVM IR: identical on both paths, so LLVM-HOST
// checks both. The only difference is where the second function's helper
// functions end up relative to the first function's, checked below under
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

// `target teams distribute` stays a generic (non-SPMD) kernel_type, but
// still needs its trip count host_eval'd.
void explicit_target_teams_distribute() {
#pragma omp target
  {
#pragma omp teams
    {
#pragma omp distribute
      for (int i = 0; i < 10; i++) {
        during(i);
      }
    }
  }
}

// CIR-HOST: cir.func{{.*}}@explicit_target_teams_distribute
// CIR-HOST: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-HOST: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
// CIR-HOST: %[[STEP_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[STEP:.*]] = cir.builtin_int_cast %[[STEP_CIR]] : !s32i -> i32
// CIR-HOST: omp.target kernel_type(generic) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[STEP]] -> %[[ASTEP:.*]] : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>
// CIR-HOST:     omp.distribute private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest (%{{.*}}) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[ASTEP]]) {
// CIR-HOST:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-HOST:          cir.call @{{.*}}during
// CIR-HOST:          omp.yield
// CIR-HOST:        }
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@explicit_target_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CIR-DEVICE:          cir.store align(4) %{{.*}}, %[[I_PRIV]] : !s32i, !cir.ptr<!s32i>
// CIR-DEVICE:          cir.call @{{.*}}during
// CIR-DEVICE:          omp.yield
// CIR-DEVICE:        }
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// The combined `target teams distribute` form decomposes into the same
// target/teams/distribute directive nesting as above, so it lowers
// identically.
void target_teams_distribute() {
#pragma omp target teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_distribute
// CIR-HOST: omp.target kernel_type(generic) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

void target_teams_split_distribute() {
#pragma omp target teams
#pragma omp distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_teams_split_distribute
// CIR-HOST: omp.target kernel_type(generic) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_teams_split_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// A split `target`, with a separately, explicitly nested but combined
// `teams distribute`, is the fourth and final form: the other mix of
// combined/split not yet covered above. Just like the other three
// forms, its omp.target still needs its trip count host_eval'd, even
// though the nested construct is now a single `teams_distribute` AST node
// rather than two separate `teams`/`distribute` nodes.
void target_split_teams_distribute() {
#pragma omp target
#pragma omp teams distribute
  for (int i = 0; i < 10; i++) {
    during(i);
  }
}

// CIR-HOST: cir.func{{.*}}@target_split_teams_distribute
// CIR-HOST: omp.target kernel_type(generic) host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}}, %{{.*}} -> %{{.*}} : i32, i32, i32) {
// CIR-HOST:   omp.teams {
// CIR-HOST:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-HOST-NEXT:  omp.loop_nest
// CIR-HOST:      }
// CIR-HOST:      omp.terminator
// CIR-HOST:    } {omp.combined}
// CIR-HOST:    omp.terminator
// CIR-HOST:  } {omp.combined}

// CIR-DEVICE: cir.func{{.*}}@target_split_teams_distribute
// CIR-DEVICE: omp.target kernel_type(generic) {
// CIR-DEVICE:   omp.teams {
// CIR-DEVICE:     omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}} : !cir.ptr<!s32i>) {
// CIR-DEVICE-NEXT:  omp.loop_nest
// CIR-DEVICE:      }
// CIR-DEVICE:      omp.terminator
// CIR-DEVICE:    } {omp.combined}
// CIR-DEVICE:    omp.terminator
// CIR-DEVICE:  } {omp.combined}

// LLVM-HOST-LABEL: define {{.*}} void @explicit_target_teams_distribute(
// LLVM-HOST:         %[[TRIPCOUNT_GEP:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST:         store i64 10, ptr %[[TRIPCOUNT_GEP]], align 8
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:       omp_offload.failed:
// LLVM-HOST:         call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_distribute(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:       omp_offload.failed:
// LLVM-HOST-CIR:         call void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams_split_distribute(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:       omp_offload.failed:
// LLVM-HOST-CIR:         call void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}(ptr null)

// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_split_teams_distribute(
// LLVM-HOST-CIR:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-CIR:         store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:       omp_offload.failed:
// LLVM-HOST-CIR:         call void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}(ptr null)

// Outlined target-entry functions: forks the teams region via
// __kmpc_fork_teams, same as the standalone `teams distribute` case in
// teams-distribute.c. CIR and OGCG just name the forked function
// differently.
//
// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID]])
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_distribute(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP2:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP2]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:       omp_offload.failed:
// LLVM-HOST-OGCG:         call void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}})

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}(
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(
// LLVM-HOST-OGCG-NEXT:    ret void

// `target_teams_split_distribute` lowers to the exact same shape as
// `target_teams_distribute` above on both the CIR and the OGCG paths; we
// check the full CFG again here instead of just the call sequence, so a
// future change that alters this shape for only this third (`target
// teams` combined, `distribute` split) form still gets caught. On the
// OGCG path, its host wrapper is emitted right after
// `target_teams_distribute`'s own outlined helper (see the comment on
// LLVM-HOST-CIR above for the CIR path's different ordering).
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams_split_distribute(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP3:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP3]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:       omp_offload.failed:
// LLVM-HOST-OGCG:         call void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID3:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID3]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID3]])
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}(
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(
// LLVM-HOST-OGCG-NEXT:    ret void

// `target_split_teams_distribute` lowers to the exact same shape as the
// other three forms above, on both the CIR and the OGCG paths.
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_split_teams_distribute(
// LLVM-HOST-OGCG:         %[[TRIPCOUNT_GEP4:.*]] = getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr %kernel_args, i32 0, i32 8
// LLVM-HOST-OGCG:         store i64 10, ptr %[[TRIPCOUNT_GEP4]], align 8
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:       omp_offload.failed:
// LLVM-HOST-OGCG:         call void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}(ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_l
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 0, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}})

// LLVM-HOST-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}(
// LLVM-HOST-CIR:         %[[TID4:omp_global_thread_num.*]] = call i32 @__kmpc_global_thread_num(ptr {{.*}})
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_init_4u(
// LLVM-HOST-CIR-SAME:      ptr {{.*}}, i32 %[[TID4]], i32 92,
// LLVM-HOST-CIR-SAME:      ptr %p.lastiter, ptr %p.lowerbound, ptr %p.upperbound, ptr %p.stride,
// LLVM-HOST-CIR-SAME:      i32 1, i32 0)
// LLVM-HOST-CIR:         %omp_loop.iv = phi i32 [ 0, %omp_loop.preheader ], [ %omp_loop.next, %omp_loop.inc ]
// LLVM-HOST-CIR:         %omp_loop.cmp = icmp ult i32 %omp_loop.iv, %{{.*}}
// LLVM-HOST-CIR-NEXT:    br i1 %omp_loop.cmp, label %omp_loop.body, label %omp_loop.exit
// LLVM-HOST-CIR:       omp_loop.exit:
// LLVM-HOST-CIR-NEXT:    call void @__kmpc_for_static_fini(ptr {{.*}}, i32 %[[TID4]])
// LLVM-HOST-CIR:       omp_loop.body:
// LLVM-HOST-CIR:         call void @during(i32

// LLVM-HOST-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}(
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_init_4(
// LLVM-HOST-OGCG:         call void @during(i32
// LLVM-HOST-OGCG:         call void @__kmpc_for_static_fini(
// LLVM-HOST-OGCG-NEXT:    ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// user_code.entry/worker.exit split are identical on both paths.
//
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:

// CIR's device-side `distribute` lowering uses the loop-callback-based
// __kmpc_distribute_static_loop_4u API, which needs an extra level of
// outlining for the per-iteration callback (teams -> distribute -> loop
// body); OGCG's distribute lowering uses the same direct init/fini API as
// on the host, needing only one level of outlining (teams -> loop body).
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}..omp_par

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_teams_distribute_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}..omp_par

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// `target_teams_split_distribute` lowers to the exact same shape as
// `target_teams_distribute` above on device too.
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_teams_split_distribute_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}..omp_par

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_split_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(

// `target_split_teams_distribute` lowers to the exact same shape as the
// other three forms above on device too.
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_split_teams_distribute_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE-CIR:     call void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-OGCG:    call void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-CIR-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}..omp_par{{.*}}(
// LLVM-DEVICE-CIR:         call void @__kmpc_distribute_static_loop_4u(ptr {{.*}}, ptr @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}..omp_par

// LLVM-DEVICE-OGCG-LABEL: define internal void @__omp_offloading_{{.*}}_target_split_teams_distribute_l{{.*}}_omp_outlined(
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_init_4(
// LLVM-DEVICE-OGCG:         call void @during(i32
// LLVM-DEVICE-OGCG:         call void @__kmpc_distribute_static_fini(
