// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

// Both sides must also translate to LLVM IR.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fopenmp-host-ir-file-path %t-host.bc -fclangir -emit-llvm %s -o /dev/null

int getLB(void);
int getStep(void);
void work(int);

// The lower bound is stored in a Sema-generated `.capture_expr.` variable.
// The host evaluates it to compute the trip count for host_eval, and the loop
// inside the omp.target evaluates it again into its own copy, since the
// induction variable update can't use the host's copy across the isolated
// region. Classic codegen also evaluates it on both sides.
void dynamic_lb() {
#pragma omp target
#pragma omp parallel
#pragma omp for
  for (int i = getLB(); i < 10; ++i)
    work(i);
}

// CIR-HOST-LABEL: cir.func{{.*}}@dynamic_lb
// CIR-HOST:         %[[HOST_LB:.*]] = cir.alloca ".capture_expr." align(4) init : !cir.ptr<!s32i>
// CIR-HOST:         %[[HOST_LB_VAL:.*]] = cir.call @getLB()
// CIR-HOST-NEXT:    cir.store align(4) %[[HOST_LB_VAL]], %[[HOST_LB]]
// CIR-HOST:         %[[TRIPCOUNT_SEL:.*]] = cir.select if
// CIR-HOST:         %[[TRIPCOUNT:.*]] = cir.builtin_int_cast %[[TRIPCOUNT_SEL]] : !u32i -> i32
// CIR-HOST:         omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[AZERO:.*]], %[[TRIPCOUNT]] -> %[[ATRIPCOUNT:.*]], %{{.*}} -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:      omp.parallel {
// CIR-HOST-NEXT:        %[[LB:.*]] = cir.alloca ".capture_expr." align(4) init : !cir.ptr<!s32i>
// CIR-HOST:             %[[LB_VAL:.*]] = cir.call @getLB()
// CIR-HOST-NEXT:        cir.store align(4) %[[LB_VAL]], %[[LB]]
// CIR-HOST:             omp.loop_nest (%{{.*}}) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST:               cir.load align(4) %[[LB]] : !cir.ptr<!s32i>, !s32i
// CIR-HOST:               cir.call @work(

// CIR-DEVICE-LABEL: cir.func{{.*}}@dynamic_lb
// CIR-DEVICE:         omp.target kernel_type(spmd) {
// CIR-DEVICE:           cir.call @getLB()
// CIR-DEVICE:           omp.loop_nest

// A non-constant step is likewise stored in a `.new_step` variable that both
// the trip count and the induction variable update read.
void dynamic_step() {
#pragma omp target parallel for
  for (int i = 0; i < 10; i += getStep())
    work(i);
}

// CIR-HOST-LABEL: cir.func{{.*}}@dynamic_step
// CIR-HOST:         %[[HOST_STEP:.*]] = cir.alloca ".new_step" align(4) init : !cir.ptr<!s32i>
// CIR-HOST:         %[[HOST_STEP_VAL:.*]] = cir.call @getStep()
// CIR-HOST-NEXT:    cir.store align(4) %[[HOST_STEP_VAL]], %[[HOST_STEP]]
// CIR-HOST:         omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[AZERO:.*]], %{{.*}} -> %[[ATRIPCOUNT:.*]], %{{.*}} -> %[[AONE:.*]] : i32, i32, i32) {
// CIR-HOST-NEXT:      omp.parallel {
// CIR-HOST-NEXT:        %[[STEP:.*]] = cir.alloca ".new_step" align(4) init : !cir.ptr<!s32i>
// CIR-HOST:             %[[STEP_VAL:.*]] = cir.call @getStep()
// CIR-HOST-NEXT:        cir.store align(4) %[[STEP_VAL]], %[[STEP]]
// CIR-HOST:             omp.loop_nest (%{{.*}}) : i32 = (%[[AZERO]]) to (%[[ATRIPCOUNT]]) step (%[[AONE]]) {
// CIR-HOST:               cir.load align(4) %[[STEP]] : !cir.ptr<!s32i>, !s32i
// CIR-HOST:               cir.call @work(

// CIR-DEVICE-LABEL: cir.func{{.*}}@dynamic_step
// CIR-DEVICE:         omp.target kernel_type(spmd) {
// CIR-DEVICE:           cir.call @getStep()
// CIR-DEVICE:           omp.loop_nest

// A parallel region without a worksharing loop isn't a target SPMD
// construct, so the omp.target is a generic kernel without host_eval, just
// like the combined `target parallel`.
void parallel_without_loop() {
#pragma omp target
#pragma omp parallel
  { work(0); }
}

// CIR-HOST-LABEL: cir.func{{.*}}@parallel_without_loop
// CIR-HOST-NEXT:    omp.target kernel_type(generic) {
// CIR-HOST-NEXT:      omp.parallel {
// CIR-HOST:             cir.call @work(

// CIR-DEVICE-LABEL: cir.func{{.*}}@parallel_without_loop
// CIR-DEVICE-NEXT:    omp.target kernel_type(generic) {
// CIR-DEVICE-NEXT:      omp.parallel {
// CIR-DEVICE:             cir.call @work(
