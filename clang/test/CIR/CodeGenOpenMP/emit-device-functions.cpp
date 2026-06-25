// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device \
// RUN:   -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Test device-side filtering of globals, mirroring CGOpenMPRuntime logic:
// - declare target functions and their transitive callees are emitted
// - host-only functions are filtered out
// - device_type(host) functions are filtered out on the device
// - host functions containing target regions are kept

// ---- declare target block: emitted on device ----

#pragma omp declare target

void regular_func() {}

struct S {
  int x;
  S() : x(42) {}
  ~S() {}
};

void caller() {
  regular_func();
  S s;
}

#pragma omp end declare target

// CIR-DAG: cir.func {{.*}} @_Z12regular_funcv() {{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)
// CIR-DAG: cir.func {{.*}} @_Z6callerv() {{.*}}omp.declare_target
// CIR-DAG: cir.func {{.*}} @_ZN1SC2Ev({{.*}})
// CIR-DAG: cir.func {{.*}} @_ZN1SC1Ev({{.*}})
// CIR-DAG: cir.func {{.*}} @_ZN1SD2Ev({{.*}})
// CIR-DAG: cir.func {{.*}} @_ZN1SD1Ev({{.*}})

// LLVM-DAG: define {{.*}} void @_Z12regular_funcv()
// LLVM-DAG: define {{.*}} void @_Z6callerv()
// LLVM-DAG: define {{.*}} void @_ZN1SC2Ev(
// LLVM-DAG: define {{.*}} void @_ZN1SC1Ev(
// LLVM-DAG: define {{.*}} void @_ZN1SD2Ev(
// LLVM-DAG: define {{.*}} void @_ZN1SD1Ev(

// OGCG-DAG: define {{.*}} void @_Z12regular_funcv()
// OGCG-DAG: define {{.*}} void @_Z6callerv()
// OGCG-DAG: define {{.*}} void @_ZN1SC2Ev(
// OGCG-DAG: define {{.*}} void @_ZN1SC1Ev(
// OGCG-DAG: define {{.*}} void @_ZN1SD2Ev(
// OGCG-DAG: define {{.*}} void @_ZN1SD1Ev(

// ---- host-only function: NOT emitted on device ----

void host_only_func() {}

// CIR-NOT: @_Z14host_only_funcv
// LLVM-NOT: @_Z14host_only_funcv
// OGCG-NOT: @_Z14host_only_funcv

// ---- device_type(host): filtered out on device by isAssumedToBeNotEmitted ----

void host_device_type_func() {}
#pragma omp declare target to(host_device_type_func) device_type(host)

// CIR-NOT: @_Z20host_device_type_funcv
// LLVM-NOT: @_Z20host_device_type_funcv
// OGCG-NOT: @_Z20host_device_type_funcv

// ---- transitive callee: emitted because called from declare target ----

void transitive_callee() {}

#pragma omp declare target
void calls_transitive() { transitive_callee(); }
#pragma omp end declare target

// CIR-DAG: cir.func {{.*}} @_Z17transitive_calleev()
// CIR-DAG: cir.func {{.*}} @_Z16calls_transitivev() {{.*}}omp.declare_target

// LLVM-DAG: define {{.*}} void @_Z17transitive_calleev()
// LLVM-DAG: define {{.*}} void @_Z16calls_transitivev()

// OGCG-DAG: define {{.*}} void @_Z17transitive_calleev()
// OGCG-DAG: define {{.*}} void @_Z16calls_transitivev()
