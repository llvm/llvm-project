// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// Check that we do not generate callee_type metadata for indirect calls
// to functions with internal linkage (e.g., types in anonymous namespaces),
// as their type metadata identifiers are distinct MDNodes instead of 
// generalized strings, which would fail the LLVM Verifier.

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

// CHECK-LABEL: define {{.*}} void @{{.*}}1a1dEv
// CHECK:   %[[VFN:.*]] = getelementptr inbounds ptr, ptr %{{.*}}, i{{[0-9]+}} 0
// CHECK:   %[[FP:.*]] = load ptr, ptr %[[VFN]], align {{[0-9]+}}
// CHECK:   call void %[[FP]]({{.*}})
// CHECK-NOT: !callee_type
// CHECK:   ret void
