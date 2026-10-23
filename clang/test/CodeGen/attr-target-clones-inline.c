// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=ATTR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=OPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=ATTR
// RUN: %clang_cc1 -triple x86_64-linux-gnu -x c++ -O2 -emit-llvm %s -o - | FileCheck %s --check-prefixes=OPT,CXX
// RUN: %clang_cc1 -triple x86_64-windows-pc -O1 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

// WINDOWS-NOT: "fmv-features"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((target_clones("default,sse4.2,avx2")))
int callee(int x) { return x + 1; }

// Reordering versions and calling through a local pointer still allow inlining.
__attribute__((target_clones("avx2,default,sse4.2")))
int caller(int x) {
  int (*p)(int) = callee;
  return p(x);
}

// OPT-LABEL: define{{.*}} i32 @caller.avx2.
// OPT-NOT: call
// OPT: add nsw i32
// OPT: ret i32
// OPT-LABEL: define{{.*}} i32 @caller.default.
// OPT-NOT: call
// OPT: add nsw i32
// OPT: ret i32
// OPT-LABEL: define{{.*}} i32 @caller.sse4.2.
// OPT-NOT: call
// OPT: add nsw i32
// OPT: ret i32

// The runtime resolver can still choose callee.avx2 from either caller here.
__attribute__((target_clones("default,sse4.2")))
int limited(int x) { return callee(x); }

// OPT-LABEL: define{{.*}} i32 @limited.default.
// OPT: call i32 @callee(
// OPT-LABEL: define{{.*}} i32 @limited.sse4.2.
// OPT: call i32 @callee(

__attribute__((target_clones("default,arch=x86-64-v3")))
double widen(_Float16 x) { return (double)x; }

__attribute__((target_clones("default,arch=x86-64-v3")))
double wide_caller(_Float16 x) { return widen(x); }

// OPT-LABEL: define{{.*}} double @wide_caller.default.
// OPT-NOT: call
// OPT: fpext half
// OPT: ret double
// OPT-LABEL: define{{.*}} double @wide_caller.arch_x86-64-v3.
// OPT-NOT: call
// OPT: fpext half
// OPT: ret double

// CPU-model predicates are deliberately left to the runtime resolver.
__attribute__((target_clones("default,arch=haswell")))
int cpu_callee(int x) { return x + 2; }

__attribute__((target_clones("default,arch=haswell")))
int cpu_caller(int x) { return cpu_callee(x); }

// OPT-LABEL: define{{.*}} i32 @cpu_caller.default.
// OPT: call i32 @cpu_callee(
// OPT-LABEL: define{{.*}} i32 @cpu_caller.arch_haswell.
// OPT: call i32 @cpu_callee(

#ifdef __cplusplus
}
// There is no enclosing FunctionDecl while emitting a global initializer.
int dynamic_init = callee(0);
// CXX-LABEL: define internal void @_GLOBAL__sub_I_
// CXX: call i32 @callee(
#endif

// Dispatch predicates, rather than the expanded ISA features, describe clones.
// ATTR-DAG: attributes #{{[0-9]+}} = { {{.*}}"fmv-features" {{.*}} }
// ATTR-DAG: attributes #{{[0-9]+}} = { {{.*}}"fmv-features"="sse4.2" {{.*}} }
// ATTR-DAG: attributes #{{[0-9]+}} = { {{.*}}"fmv-features"="avx2" {{.*}} }
// ATTR-DAG: attributes #{{[0-9]+}} = { {{.*}}"fmv-features"="x86-64-v3" {{.*}} }
