// Host compilation (x86 host, AMDGPU offload target).
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas live in the private address space.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void use(int);

// The combined 'target parallel' directive lowers to an omp.parallel nested
// inside an omp.target, identical to the equivalent nesting of the separate
// 'target' and 'parallel' directives.
void target_parallel(int x) {
  // CIR-HOST: cir.func{{.*}}@target_parallel
  // CIR-HOST: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} {name = "x"}
  // CIR-HOST: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.parallel {
  // CIR-HOST: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST: cir.call @use(%[[LOAD]])
  // CIR-HOST: omp.terminator
  // CIR-HOST: }
  // CIR-HOST: omp.terminator
  // CIR-HOST: }

  // CIR-DEVICE: cir.func{{.*}}@target_parallel
  // CIR-DEVICE: omp.target kernel_type(generic) {{.*}} {
  // CIR-DEVICE: omp.parallel {
  // CIR-DEVICE: cir.call @use
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE: }
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
  // CIR-HOST: }
#pragma omp target parallel proc_bind(spread) map(tofrom : x)
  {
    use(x);
  }
}
