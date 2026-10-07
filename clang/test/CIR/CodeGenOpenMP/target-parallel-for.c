// REQUIRES: amdgpu-registered-target

// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void during(int);

// The legal nesting of target, parallel and for lowers to an omp.wsloop +
// omp.loop_nest inside omp.parallel inside omp.target. The worksharing loop
// bounds and induction variable are cast between CIR and builtin integers
// with cir.builtin_int_cast on both the host and the GPU device.
//
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

// The trip count is computed from the original bounds/step via Sema's
// helper expression, then 0, the trip count, and 1 are each cast to a
// builtin integer and forwarded as the omp.target's host_eval operands.
// CIR-HOST: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-HOST: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-HOST: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-HOST: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32
// CIR-HOST: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-HOST: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%[[ZERO]] -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %[[ONE]] -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST: omp.parallel {

// The normalized 0-based counter and the real induction variable each get
// their own alloca before the wsloop; "i"'s alloca has no `init` because
// its value is now produced by the update expression below, not by
// directly emitting the for-statement's init.
// CIR-HOST: %[[IV_ALLOCA:.*]] = cir.alloca ".omp.iv" align(4) : !cir.ptr<!s32i>
// CIR-HOST: %[[I_ALLOCA:.*]] = cir.alloca "i" align(4) : !cir.ptr<!s32i>

// "i" is registered as predetermined-private on the wsloop's `private`
// clause (see parallel-for.c for why %[[I_ALLOCA]] is never read).
// CIR-HOST: omp.wsloop private(@{{.*}} %[[I_ALLOCA]] -> %[[I_PRIV:.*]] : !cir.ptr<!s32i>) {
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

// The device recomputes the same normalized `[0, tripCount)` bounds itself
// (no host_eval here -- see the file comment above).
// CIR-DEVICE: %[[TRIPCOUNT_CIR:.*]] = cir.div %{{.*}}, %{{.*}} : !s32i
// CIR-DEVICE: %[[ZERO_CIR:.*]] = cir.const #cir.int<0> : !s32i
// CIR-DEVICE: %[[ZERO:.*]] = cir.builtin_int_cast %[[ZERO_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[ONE_CIR:.*]] = cir.const #cir.int<1> : !s32i
// CIR-DEVICE: %[[ONE:.*]] = cir.builtin_int_cast %[[ONE_CIR]] : !s32i -> i32
// CIR-DEVICE: %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_CIR]] : !s32i -> i32

// "i" is registered as predetermined-private on the wsloop's `private`
// clause (see parallel-for.c for why %[[I_CAST]] is never read).
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

// The `target` and `parallel` are non-innermost leaves of the combined
// construct, so both carry the omp.combined attribute, just like the
// explicitly nested directives above.
// CIR-HOST: cir.func{{.*}}@combined_target_parallel_for
// CIR-HOST: omp.target kernel_type(spmd) host_eval(%[[ZERO:.*]] -> %[[AZERO:.*]], %[[TRIPCOUNT:.*]] -> %[[ATRIPCOUNT:.*]], %[[ONE:.*]] -> %[[AONE:.*]] : i32, i32, i32) {
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
