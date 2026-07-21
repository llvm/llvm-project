// Host compilation (x86 host, AMDGPU offload target): no address space on allocas.
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-HOST

// Device compilation (AMDGPU): allocas in private address space, addrspacecast for map info.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-cir -fclangir %s -o - \
// RUN:   | FileCheck %s --check-prefix=CIR-DEVICE

void use(int);

void target_map_to(int x) {
  // CIR-HOST: cir.func{{.*}}@target_map_to
  // CIR-HOST: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
  // CIR-HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-HOST-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST-NEXT: %[[LOAD:.*]] = cir.load align(4) %[[ARG]]
  // CIR-HOST-NEXT: cir.call @use(%[[LOAD]])
  // CIR-HOST-NEXT: omp.terminator
  // CIR-HOST-NEXT: }

  // CIR-DEVICE: cir.func{{.*}}@target_map_to
  // CIR-DEVICE: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
  // CIR-DEVICE: %[[CAST:.*]] = cir.cast address_space %[[X_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
  // CIR-DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-DEVICE-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE-NEXT: }
#pragma omp target map(to : x)
  {
    use(x);
  }
}

void target_map_from(int x) {
  // CIR-HOST: cir.func{{.*}}@target_map_from
  // CIR-HOST: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
  // CIR-HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-HOST-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST-NEXT: %[[C42:.*]] = cir.const #cir.int<42> : !s32i
  // CIR-HOST-NEXT: cir.store align(4) %[[C42]], %[[ARG]]
  // CIR-HOST-NEXT: omp.terminator
  // CIR-HOST-NEXT: }

  // CIR-DEVICE: cir.func{{.*}}@target_map_from
  // CIR-DEVICE: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
  // CIR-DEVICE: %[[CAST:.*]] = cir.cast address_space %[[X_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
  // CIR-DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-DEVICE-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE-NEXT: }
#pragma omp target map(from : x)
  {
    x = 42;
  }
}

void target_map_tofrom(int x) {
  // CIR-HOST: cir.func{{.*}}@target_map_tofrom
  // CIR-HOST: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
  // CIR-HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-HOST-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-HOST: omp.terminator
  // CIR-HOST-NEXT: }

  // CIR-DEVICE: cir.func{{.*}}@target_map_tofrom
  // CIR-DEVICE: %[[X_ALLOCA:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
  // CIR-DEVICE: %[[CAST:.*]] = cir.cast address_space %[[X_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
  // CIR-DEVICE: %[[MAP:.*]] = omp.map.info var_ptr(%[[CAST]] : !cir.ptr<!s32i>, !s32i) map_clauses(tofrom) capture(ByRef) -> !cir.ptr<!s32i> {name = "x"}
  // CIR-DEVICE-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP]] -> %[[ARG:.*]] : !cir.ptr<!s32i>) {
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE-NEXT: }
#pragma omp target map(tofrom : x)
  {
    x = x + 1;
  }
}

void target_map_multiple(int a, int b) {
  // CIR-HOST: cir.func{{.*}}@target_map_multiple
  // CIR-HOST-DAG: %[[A_ALLOCA:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
  // CIR-HOST-DAG: %[[B_ALLOCA:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
  // CIR-HOST: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[A_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "a"}
  // CIR-HOST-NEXT: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[B_ALLOCA]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "b"}
  // CIR-HOST-NEXT: omp.target kernel_type(generic) map_entries(%[[MAP_A]] -> %[[ARG_A:.*]], %[[MAP_B]] -> %[[ARG_B:.*]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CIR-HOST: omp.terminator
  // CIR-HOST-NEXT: }

  // CIR-DEVICE: cir.func{{.*}}@target_map_multiple
  // CIR-DEVICE-DAG: %[[A_ALLOCA:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
  // CIR-DEVICE-DAG: %[[B_ALLOCA:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i, target_address_space(5)>
  // CIR-DEVICE: %[[CAST_A:.*]] = cir.cast address_space %[[A_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
  // CIR-DEVICE: %[[MAP_A:.*]] = omp.map.info var_ptr(%[[CAST_A]] : !cir.ptr<!s32i>, !s32i) map_clauses(to) capture(ByRef) -> !cir.ptr<!s32i> {name = "a"}
  // CIR-DEVICE: %[[CAST_B:.*]] = cir.cast address_space %[[B_ALLOCA]] : !cir.ptr<!s32i, target_address_space(5)> -> !cir.ptr<!s32i>
  // CIR-DEVICE: %[[MAP_B:.*]] = omp.map.info var_ptr(%[[CAST_B]] : !cir.ptr<!s32i>, !s32i) map_clauses(from) capture(ByRef) -> !cir.ptr<!s32i> {name = "b"}
  // CIR-DEVICE: omp.target kernel_type(generic) map_entries(%[[MAP_A]] -> %[[ARG_A:.*]], %[[MAP_B]] -> %[[ARG_B:.*]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CIR-DEVICE: omp.terminator
  // CIR-DEVICE-NEXT: }
#pragma omp target map(to : a) map(from : b)
  {
    b = a;
  }
}

// TODO: Test implicit mapping. Currently NYI
