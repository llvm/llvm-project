// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=ATTR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefixes=ATTR,INIT
// RUN: %clang_cc1 -triple x86_64-windows-pc -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

// WINDOWS-NOT: "fmv-features"

// ATTR: define{{.*}} i32 @callee.default.2({{.*}}) #[[DEFAULT:[0-9]+]]
// ATTR: define{{.*}} i32 @callee.sse4.2.0({{.*}}) #[[SSE42:[0-9]+]]
// ATTR: define{{.*}} i32 @callee.avx2.1({{.*}}) #[[AVX2:[0-9]+]]
// ATTR: define{{.*}} i32 @caller.avx2.0({{.*}}) #[[AVX2]]
// ATTR: define{{.*}} i32 @caller.default.2({{.*}}) #[[DEFAULT]]
// ATTR: define{{.*}} i32 @caller.sse4.2.1({{.*}}) #[[SSE42]]
// ATTR: define{{.*}} i32 @limited.default.1({{.*}}) #[[DEFAULT]]
// ATTR: define{{.*}} i32 @limited.sse4.2.0({{.*}}) #[[SSE42]]
// ATTR: define{{.*}} double @widen.default.1({{.*}}) #[[DEFAULT]]
// ATTR: define{{.*}} double @widen.arch_x86-64-v3.0({{.*}}) #[[V3:[0-9]+]]
// ATTR: define{{.*}} double @wide_caller.default.1({{.*}}) #[[DEFAULT]]
// ATTR: define{{.*}} double @wide_caller.arch_x86-64-v3.0({{.*}}) #[[V3]]
// ATTR: define{{.*}} i32 @cpu_callee.default.1({{.*}}) #[[CPU_DEFAULT:[0-9]+]]
// ATTR: define{{.*}} i32 @cpu_callee.arch_haswell.0({{.*}}) #[[CPU:[0-9]+]]
// ATTR: define{{.*}} i32 @cpu_caller.default.1({{.*}}) #[[CPU_DEFAULT]]
// ATTR: define{{.*}} i32 @cpu_caller.arch_haswell.0({{.*}}) #[[CPU]]

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((target_clones("default,sse4.2,avx2")))
int callee(int x) { return x + 1; }

// Attribute values follow each clone, including reordered version lists.
__attribute__((target_clones("avx2,default,sse4.2")))
int caller(int x) {
  int (*p)(int) = callee;
  return p(x);
}

// A caller with fewer versions still gets its own dispatch predicates.
__attribute__((target_clones("default,sse4.2")))
int limited(int x) { return callee(x); }

__attribute__((target_clones("default,arch=x86-64-v3")))
double widen(_Float16 x) { return (double)x; }

__attribute__((target_clones("default,arch=x86-64-v3")))
double wide_caller(_Float16 x) { return widen(x); }

// CPU-model predicates retain CPU identity and the exact resolver priority.
__attribute__((target_clones("default,arch=haswell")))
int cpu_callee(int x) { return x + 2; }

__attribute__((target_clones("default,arch=haswell")))
int cpu_caller(int x) { return cpu_callee(x); }

#ifdef __cplusplus
}
// A global initializer must call the runtime-dispatched function, since it
// has no selected clone whose features could determine the callee version.
// INIT-LABEL: define internal void @__cxx_global_var_init(
// INIT: call i32 @callee(
int dynamic_init = callee(0);
#endif

// Dispatch predicates describe the runtime selection of each clone.
// ATTR: attributes #[[DEFAULT]] = { {{.*}}"fmv-features" {{.*}} }
// ATTR: attributes #[[SSE42]] = { {{.*}}"fmv-features"="sse4.2" {{.*}} }
// ATTR: attributes #[[AVX2]] = { {{.*}}"fmv-features"="avx2" {{.*}} }
// ATTR: attributes #[[V3]] = { {{.*}}"fmv-features"="x86-64-v3" {{.*}} }
// ATTR: attributes #[[CPU_DEFAULT]] = { {{.*}}"fmv-features" "fmv-priority"="{{[0-9]+}}"
// ATTR: attributes #[[CPU]] = { {{.*}}"fmv-features"="arch=haswell" "fmv-priority"="{{[0-9]+}}"
