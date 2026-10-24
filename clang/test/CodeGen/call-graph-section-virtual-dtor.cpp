// Tests that callee_type metadata is attached to indirect calls that are
// emitted without asking CodeGenFunction::EmitCall for the emitted call
// instruction, such as virtual destructor calls.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -emit-llvm -o - %s | FileCheck %s

struct Base {
  virtual ~Base() {}
  virtual void vf() {}
};

struct Derived : Base {
  ~Derived() override {}
  void vf() override {}
};

// CHECK-LABEL: define {{.*}} @_Z9call_dtorP4Base(
// CHECK: call void %{{.*}}, !callee_type [[DTOR_CT:![0-9]+]]
void call_dtor(Base *b) { delete b; }

// CHECK-LABEL: define {{.*}} @_Z7call_vfP4Base(
// CHECK: call void %{{.*}}, !callee_type [[DTOR_CT]]
void call_vf(Base *b) { b->vf(); }

Base *make() { return new Derived(); }

// CHECK-DAG: define {{.*}} @_ZN7DerivedD0Ev({{.*}} !callgraph [[DTOR:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN7Derived2vfEv({{.*}} !callgraph [[DTOR]] {

// CHECK-DAG: [[DTOR]] = !{!"_ZTSFvvE"}
// CHECK-DAG: [[DTOR_CT]] = !{[[DTOR]]}
