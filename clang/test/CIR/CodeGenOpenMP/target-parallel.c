// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Host-side lowering to LLVM IR: identical on both paths, so LLVM-HOST
// checks both. The only difference is where the two helper functions end
// up relative to each other, checked below under
// LLVM-HOST-CIR/LLVM-HOST-OGCG.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefixes=LLVM-HOST,LLVM-HOST-OGCG

// Device-side lowering to LLVM IR: compiled in two steps to mirror the
// offloading driver. Same LLVM-DEVICE/LLVM-DEVICE-CIR/LLVM-DEVICE-OGCG
// split as above.
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

void use(int);

// The combined 'target parallel' directive lowers to an omp.parallel nested
// inside an omp.target, identical to the equivalent nesting of the separate
// 'target' and 'parallel' directives.
void target_parallel(int x) {
  // CIR-HOST: cir.func{{.*}}@target_parallel
  // CIR-HOST: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("x")
  // CIR-HOST: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.parallel {
  // CIR-HOST: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST: cir.call @use(%[[LOAD]])
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // The target is the non-innermost leaf of the combined construct.
  // CIR-HOST: } {omp.combined}

  // CIR-DEVICE: cir.func{{.*}}@target_parallel
  // CIR-DEVICE: omp.target kernel_type(generic) {{.*}} {
  // CIR-DEVICE: omp.parallel {
  // CIR-DEVICE: cir.call @use
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: } {omp.combined}
#pragma omp target parallel map(tofrom : x)
  {
    use(x);
  }
}

// 'target parallel' routes the proc_bind clause to the parallel leaf and the
// map clause to the target leaf.
void target_parallel_proc_bind(int x) {
  // CIR-HOST: cir.func{{.*}}@target_parallel_proc_bind
  // CIR-HOST: omp.target kernel_type(generic) map_entries({{.*}}) {
  // CIR-HOST: omp.parallel proc_bind(spread) {
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // CIR-HOST: } {omp.combined}
#pragma omp target parallel proc_bind(spread) map(tofrom : x)
  {
    use(x);
  }
}

// Host wrappers: the __tgt_target_kernel call, the omp_offload.failed branch,
// and the call into the offloading-entry function are identical on both
// paths (the entry symbol name is derived from file+line, not from the
// lowering path).
//
// LLVM-HOST-LABEL: define {{.*}} void @target_parallel(
// LLVM-HOST-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST:         call i32 @__tgt_target_kernel(
// LLVM-HOST:       omp_offload.failed:
// LLVM-HOST:         call void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// CIR groups all its helper functions together up front, since the
// MLIR-to-LLVM translation emits them during the lowering pass itself;
// OGCG's direct AST walk emits each one in source order instead,
// interleaved with other constructs. The checks are the same either way,
// just placed where each path actually emits them (compare the
// LLVM-HOST-OGCG copy of this block further down).
// LLVM-HOST-CIR-LABEL: define {{.*}} void @target_parallel_proc_bind(
// LLVM-HOST-CIR-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST-CIR:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST-CIR:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST-CIR:         call i32 @__tgt_target_kernel(
// LLVM-HOST-CIR:       omp_offload.failed:
// LLVM-HOST-CIR:         call void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// Outlined target-entry functions: forks the parallel region via
// __kmpc_fork_call, with a __kmpc_push_proc_bind call right before it for
// the proc_bind clause. CIR and OGCG just name the forked function
// differently, which we don't check since it doesn't matter.
//
// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}, ptr {{.*}})

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(
// LLVM-HOST:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-HOST:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-HOST:         call void @use(i32 {{.*}} %[[V]])
// LLVM-HOST:         ret void

// OGCG's copy of target_parallel_proc_bind's host wrapper: identical
// content to the LLVM-HOST-CIR block above, just emitted after
// target_parallel's entry+outlined functions instead of before them.
// LLVM-HOST-OGCG-LABEL: define {{.*}} void @target_parallel_proc_bind(
// LLVM-HOST-OGCG-SAME:  i32 noundef %[[ARG:[^,)]+]]
// LLVM-HOST-OGCG:         %[[X_ADDR:.*]] = alloca i32, align 4
// LLVM-HOST-OGCG:         store i32 {{.*}}, ptr %[[X_ADDR]], align 4
// LLVM-HOST-OGCG:         call i32 @__tgt_target_kernel(
// LLVM-HOST-OGCG:       omp_offload.failed:
// LLVM-HOST-OGCG:         call void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(ptr %[[X_ADDR]], ptr null)

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(
// LLVM-HOST:         call i32 @__kmpc_global_thread_num(
// LLVM-HOST:         call void @__kmpc_push_proc_bind(ptr {{.*}}, i32 {{.*}}, i32 4)
// LLVM-HOST:         call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr {{.*}}, i32 1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}, ptr {{.*}})

// LLVM-HOST-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(
// LLVM-HOST:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-HOST:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-HOST:         call void @use(i32 {{.*}} %[[V]])
// LLVM-HOST:         ret void

// Device kernel entry: __kmpc_target_init/__kmpc_target_deinit and the
// user_code.entry/worker.exit split are identical on both paths.
//
// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE:         call i32 @__kmpc_global_thread_num(

// CIR's outlining always builds a dedicated wrapper function for the
// parallel region; OGCG's codegen only builds one when it's actually
// needed, so it passes a null function pointer here instead.
// LLVM-DEVICE-CIR:   call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}.wrapper, ptr {{.*}}, i64 1, i32 0)
// LLVM-DEVICE-OGCG:  call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_l{{.*}}, ptr null, ptr {{.*}}, i64 1, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// The forked function's body (loading the captured arg and calling @use) is
// identical on both paths.
//
// LLVM-DEVICE-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_l{{.*}}(
// LLVM-DEVICE:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-DEVICE:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-DEVICE:         call void @use(i32 {{.*}} %[[V]])
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-LABEL: define weak_odr protected amdgpu_kernel void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l
// LLVM-DEVICE:         call i32 @__kmpc_target_init(
// LLVM-DEVICE:       user_code.entry:
// LLVM-DEVICE:         call i32 @__kmpc_global_thread_num(

// LLVM-DEVICE-CIR:   call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}.wrapper, ptr {{.*}}, i64 1, i32 0)
// LLVM-DEVICE-OGCG:  call void @__kmpc_parallel_60(ptr {{.*}}, i32 {{.*}}, i32 1, i32 -1, i32 -1, ptr @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}, ptr null, ptr {{.*}}, i64 1, i32 0)

// LLVM-DEVICE:         call void @__kmpc_target_deinit()
// LLVM-DEVICE:         ret void

// LLVM-DEVICE-LABEL: define internal void @__omp_offloading_{{.*}}_target_parallel_proc_bind_l{{.*}}(
// LLVM-DEVICE:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM-DEVICE:         %[[V:.*]] = load i32, ptr %[[P]], align 4
// LLVM-DEVICE:         call void @use(i32 {{.*}} %[[V]])
// LLVM-DEVICE:         ret void
