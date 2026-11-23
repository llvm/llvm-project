// Kernels for a CPU offload target take one argument array; with debug info
// the region is outlined into a "_debug__" function the kernel calls.
//
// RUN: %clang_cc1 -fopenmp -x c++ -Wno-vla -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -Wno-vla -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CXX
// RUN: %clang_cc1 -fopenmp -x c++ -Wno-vla -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -debug-info-kind=line-tables-only -emit-llvm %s -o - | FileCheck %s --check-prefix=NOWRAP
// RUN: %clang_cc1 -fopenmp -x c++ -Wno-vla -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --check-prefix=NOWRAP
//
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-c-host.bc
// RUN: %clang_cc1 -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-c-host.bc -debug-info-kind=constructor -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK

// NOWRAP-NOT: _debug__

struct S {
  int len;
};

int main(void) {
#pragma omp target
  {}
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.+}}main_l{{[0-9]+}}_debug__(
// CHECK-SAME: ptr noalias noundef %dyn_ptr) #{{[0-9]+}} !dbg !{{[0-9]+}} {

// CHECK-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}main_l{{[0-9]+}}(
// CHECK-SAME: ptr noalias noundef %__context) #{{[0-9]+}} !dbg !{{[0-9]+}} {
// CHECK: [[CTX:%.+]] = load ptr, ptr %__context.addr,
// CHECK-NEXT: [[SLOT0:%.+]] = getelementptr inbounds ptr, ptr [[CTX]], i32 0
// CHECK-NEXT: [[ARG0:%.+]] = load ptr, ptr [[SLOT0]],
// CHECK-NEXT: [[DYN:%.+]] = load ptr, ptr [[ARG0]],
// CHECK-NEXT: call void @__omp_offloading_{{.+}}main_l{{[0-9]+}}_debug__(ptr [[DYN]])
// CHECK-NEXT: ret void

// NOWRAP-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}main_l{{[0-9]+}}(
// NOWRAP-SAME: ptr noalias noundef %__context)
// NOWRAP: %dyn_ptr.addr = getelementptr inbounds ptr, ptr {{%.+}}, i32 0
// NOWRAP: ret void

void foo() {
  struct S s1;

#pragma omp target map(tofrom : s1)
  {
    s1.len = 0;
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.+}}foo{{.*}}_l{{[0-9]+}}_debug__(
// CHECK-SAME: ptr noundef nonnull align 4 dereferenceable(4) %s1, ptr noalias noundef %dyn_ptr) #{{[0-9]+}} !dbg !{{[0-9]+}} {

// CHECK-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}foo{{.*}}_l{{[0-9]+}}(
// CHECK-SAME: ptr noalias noundef %__context) #{{[0-9]+}} !dbg !{{[0-9]+}} {
// CHECK: [[CTX:%.+]] = load ptr, ptr %__context.addr,
// CHECK: [[SLOT1:%.+]] = getelementptr inbounds ptr, ptr [[CTX]], i32 1
// CHECK-NEXT: [[ARG1:%.+]] = load ptr, ptr [[SLOT1]],
// CHECK-NEXT: [[DYN:%.+]] = load ptr, ptr [[ARG1]],
// CHECK-NEXT: call void @__omp_offloading_{{.+}}foo{{.*}}_l{{[0-9]+}}_debug__(ptr %s1{{[0-9]*}}, ptr [[DYN]])
// CHECK-NEXT: ret void

// NOWRAP-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}foo{{.*}}_l{{[0-9]+}}(
// NOWRAP: %s1 = load ptr, ptr
// NOWRAP: %dyn_ptr.addr = getelementptr inbounds ptr, ptr {{%.+}}, i32 1
// NOWRAP: store i32 0, ptr %len

void captures(int n, int *p) {
  int a = 1;
  float vla[n];
#pragma omp target firstprivate(a) map(tofrom : vla[0 : n])
  {
    vla[a] = p[0] + n;
  }
}

// CHECK-LABEL: define internal void @__omp_offloading_{{.+}}captures{{.*}}_l{{[0-9]+}}_debug__(
// CHECK-SAME: i64 noundef %vla, ptr noundef nonnull align 4 dereferenceable(4) %vla1, i32 noundef %a, ptr noundef %p, i32 noundef %n, ptr noalias noundef %dyn_ptr) #{{[0-9]+}} !dbg !{{[0-9]+}} {

// CHECK-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}captures{{.*}}_l{{[0-9]+}}(
// CHECK-SAME: ptr noalias noundef %__context) #{{[0-9]+}} !dbg !{{[0-9]+}} {
// CHECK: [[CTX:%.+]] = load ptr, ptr %__context.addr,
// CHECK: [[SLOT5:%.+]] = getelementptr inbounds ptr, ptr [[CTX]], i32 5
// CHECK-NEXT: [[ARG5:%.+]] = load ptr, ptr [[SLOT5]],
// CHECK-NEXT: [[DYN:%.+]] = load ptr, ptr [[ARG5]],
// CHECK-NEXT: call void @__omp_offloading_{{.+}}captures{{.*}}_l{{[0-9]+}}_debug__(i64 %vla{{[0-9]*}}, ptr %vla{{[0-9]*}}, i32 %a{{[0-9]*}}, ptr %p{{[0-9]*}}, i32 %n{{[0-9]*}}, ptr [[DYN]])
// CHECK-NEXT: ret void

// NOWRAP-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}captures{{.*}}_l{{[0-9]+}}(
// NOWRAP: %dyn_ptr.addr = getelementptr inbounds ptr, ptr {{%.+}}, i32 5
// NOWRAP: ret void

#ifdef __cplusplus
struct T {
  int x;
  void m() {
#pragma omp target
    { x = 1; }
  }
};

void use_t(T &t) { t.m(); }

// CXX-LABEL: define internal void @__omp_offloading_{{.+}}1T1mEv_l{{[0-9]+}}_debug__(
// CXX-SAME: ptr noundef %this, ptr noalias noundef %dyn_ptr) #{{[0-9]+}} !dbg !{{[0-9]+}} {

// CXX-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}1T1mEv_l{{[0-9]+}}(
// CXX-SAME: ptr noalias noundef %__context) #{{[0-9]+}} !dbg !{{[0-9]+}} {
// CXX: [[CTX:%.+]] = load ptr, ptr %__context.addr,
// CXX: [[SLOT1:%.+]] = getelementptr inbounds ptr, ptr [[CTX]], i32 1
// CXX-NEXT: [[ARG1:%.+]] = load ptr, ptr [[SLOT1]],
// CXX-NEXT: [[DYN:%.+]] = load ptr, ptr [[ARG1]],
// CXX-NEXT: call void @__omp_offloading_{{.+}}1T1mEv_l{{[0-9]+}}_debug__(ptr %this{{[0-9]*}}, ptr [[DYN]])
// CXX-NEXT: ret void

// NOWRAP-LABEL: define weak_odr protected void @__omp_offloading_{{.+}}1T1mEv_l{{[0-9]+}}(
// NOWRAP: %this = load ptr, ptr
// NOWRAP: %dyn_ptr.addr = getelementptr inbounds ptr, ptr {{%.+}}, i32 1
// NOWRAP: store i32 1, ptr %x
#endif

// CHECK: distinct !DISubprogram(name: "__omp_offloading_{{.+}}main_l{{[0-9]+}}_debug__"
// CHECK: distinct !DISubprogram(name: "__omp_offloading_{{.+}}main_l{{[0-9]+}}",

// NOWRAP-NOT: _debug__
