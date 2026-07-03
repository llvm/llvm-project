// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Check that we safely and correctly generate callee_type and callgraph metadata
// for indirect calls and function definitions with internal linkage (e.g., types in anonymous namespaces),
// using generalized MDString type identifiers.

namespace {
class a;
class b {
public:
  virtual void c(a);
};
class a {
public:
  b &e;
  void d() { e.c(*this); }
};

void b::c(a) {}

void f() {
  a *g = nullptr;
  g->d();
}
} // namespace

void test() {
  f();
}

// CHECK-LABEL: define internal void @_ZN12_GLOBAL__N_11a1dEv(
// CHECK-SAME: ptr noundef nonnull align 8 dereferenceable(8) %this) {{.*}} !callgraph [[F_TCLS1:![0-9]+]] {
// CHECK:   %[[VFN:.*]] = getelementptr inbounds ptr, ptr %{{.*}}, i{{[0-9]+}} 0
// CHECK:   %[[FP:.*]] = load ptr, ptr %[[VFN]], align {{[0-9]+}}
// CHECK:   call void %[[FP]]({{.*}}), !callee_type [[F_TCLS2_CT:![0-9]+]]
// CHECK:   ret void

// CHECK: [[F_TCLS1]] = !{!"_ZTSFvvE.generalized", i1 true}
// CHECK: [[F_TCLS2_CT]] = !{[[F_TCLS2:![0-9]+]]}
// CHECK: [[F_TCLS2]] = !{!"_ZTSFvN12_GLOBAL__N_11aEE.generalized"}
