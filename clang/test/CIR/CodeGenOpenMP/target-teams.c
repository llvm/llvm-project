// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU).
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
// RUN:   | FileCheck %s --check-prefix=LLVM-DEVICE
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-ogcg-host.bc \
// RUN:   -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=LLVM-DEVICE

void use(int);

// Explicit `target` + `teams` nesting lowers the same as the combined
// `target teams` spelling below, so both get omp.combined.
void explicit_target_teams(int x) {
  // CIR-HOST: cir.func{{.*}}@explicit_target_teams
  // CIR-HOST: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("x")
  // CIR-HOST: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.teams {
  // CIR-HOST: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST: cir.call @use(%[[LOAD]])
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // CIR-HOST: } {omp.combined}

  // CIR-DEVICE: cir.func{{.*}}@explicit_target_teams
  // CIR-DEVICE: omp.target kernel_type(generic) {{.*}} {
  // CIR-DEVICE: omp.teams {
  // CIR-DEVICE: cir.call @use
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: } {omp.combined}
#pragma omp target map(tofrom : x)
#pragma omp teams
  {
    use(x);
  }
}

// Combined `target teams` lowers the same as the explicit nesting above,
// with omp.combined on the omp.target.
void target_teams(int x) {
  // CIR-HOST: cir.func{{.*}}@target_teams
  // CIR-HOST: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("x")
  // CIR-HOST: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.teams {
  // CIR-HOST: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST: cir.call @use(%[[LOAD]])
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // CIR-HOST: } {omp.combined}

  // CIR-DEVICE: cir.func{{.*}}@target_teams
  // CIR-DEVICE: omp.target kernel_type(generic) {{.*}} {
  // CIR-DEVICE: omp.teams {
  // CIR-DEVICE: cir.call @use
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: } {omp.combined}
#pragma omp target teams map(tofrom : x)
  {
    use(x);
  }
}

// LLVM-HOST-LABEL: define {{.*}} void @explicit_target_teams(
// LLVM-HOST-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:         br i1 %{{.*}}, label %[[FAILED:[^,]+]], label %{{.*}}
// LLVM-HOST:       {{^}}[[FAILED]]:
// LLVM-HOST-NEXT:    call void @__omp_offloading_{{.*}}_explicit_target_teams_l{{[0-9]+}}(ptr %[[X_ADDR]], ptr null)

// CIR groups all its helper functions together up front, since the
// MLIR-to-LLVM translation emits them during the lowering pass itself;
// OGCG's direct AST walk emits each one in source order instead,
// interleaved with other constructs. The checks are the same either way,
// just placed where each path actually emits them (compare the
// LLVM-HOST-OGCG copy of this block further down).
// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_teams(
// LLVM-HOST-CIR-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST-CIR:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST-CIR:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-CIR:       {{^}}[[FAILED2]]:
// LLVM-HOST-CIR-NEXT:    call void @__omp_offloading_{{.*}}_target_teams_l{{[0-9]+}}(ptr %[[X_ADDR]], ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_explicit_target_teams_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @[[OUTLINED:[^,]+]], ptr {{.*}})

// LLVM-HOST:       define internal void @[[OUTLINED]](
// LLVM-HOST:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-HOST:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-HOST:         call void @use(i32 {{.*}} %[[V]])
// LLVM-HOST:         ret void

// OGCG's copy of target_teams's host wrapper: identical content to the
// LLVM-HOST-CIR block above, just emitted after explicit_target_teams's
// outlined functions instead of before them.
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_teams(
// LLVM-HOST-OGCG-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST-OGCG:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST-OGCG:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:         br i1 %{{.*}}, label %[[FAILED2:[^,]+]], label %{{.*}}
// LLVM-HOST-OGCG:       {{^}}[[FAILED2]]:
// LLVM-HOST-OGCG-NEXT:    call void @__omp_offloading_{{.*}}_target_teams_l{{[0-9]+}}(ptr %[[X_ADDR]], ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_teams_l{{[0-9]+}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_teams(ptr {{.*}}, i32 1, ptr @[[OUTLINED2:[^,]+]], ptr {{.*}})

// LLVM-HOST:       define internal void @[[OUTLINED2]](
// LLVM-HOST:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-HOST:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-HOST:         call void @use(i32 {{.*}} %[[V]])
// LLVM-HOST:         ret void

// Device kernel entry: after __kmpc_target_init, the user code calls the
// teams-outlined function directly; there is no fork call on the device.
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_explicit_target_teams_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE]]:
// LLVM-DEVICE:         call void @[[DEV_OUTLINED:__omp_offloading_[^(]+]](ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE:       define internal void @[[DEV_OUTLINED]](
// LLVM-DEVICE:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-DEVICE:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-DEVICE:         call void @use(i32 {{.*}} %[[V]])
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_teams_l{{[0-9]+}}(
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:         br i1 %{{.*}}, label %[[USER_CODE2:[^,]+]], label %{{.*}}
// LLVM-DEVICE:       {{^}}[[USER_CODE2]]:
// LLVM-DEVICE:         call void @[[DEV_OUTLINED2:__omp_offloading_[^(]+]](ptr {{.*}}, ptr {{.*}}, ptr {{.*}})
// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE:       define internal void @[[DEV_OUTLINED2]](
// LLVM-DEVICE:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-DEVICE:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-DEVICE:         call void @use(i32 {{.*}} %[[V]])
// LLVM-DEVICE:         ret void
