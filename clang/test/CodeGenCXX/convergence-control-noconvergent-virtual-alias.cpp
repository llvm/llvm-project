// RUN: %clang_cc1 -triple spirv1.6-unknown-vulkan1.3-compute -fconvergent-functions -mconstructor-aliases -emit-llvm -o - %s | FileCheck %s

// With convergence tokens, [[clang::noconvergent]] does not change whether a
// call is convergent. The call site attribute is only left off when the callee
// is a convergent function when the call is emitted.

struct V {
  virtual void f();
};

struct S {
  ~S();
};

// CHECK: @_ZN1SD1Ev = unnamed_addr alias void (ptr), ptr @_ZN1SD2Ev

// CHECK-LABEL: define spir_func void @_Z12call_virtualP1V(
// CHECK: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// CHECK: call spir_func addrspace(10) void %{{[0-9]+}}(ptr noundef nonnull align 8 dereferenceable(8) %{{[0-9]+}}) [[CONVERGENT:#[0-9]+]] [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
void call_virtual(V *v) {
  [[clang::noconvergent]] v->f();
}

// The call is emitted while the complete destructor is still a convergent
// function, so it gets no call site attribute. The destructor later becomes an
// alias of the base destructor, and the token alone keeps the call convergent.
// CHECK-LABEL: define spir_func void @_Z9call_dtorP1S(
// CHECK: [[TOKEN:%.*]] = call token @llvm.experimental.convergence.entry()
// CHECK: call spir_func void @_ZN1SD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %{{[0-9]+}}) [[NOUNWIND:#[0-9]+]] [ "convergencectrl"(token [[TOKEN]]) ]{{$}}
void call_dtor(S *s) {
  [[clang::noconvergent]] s->~S();
}

S::~S() {}

// CHECK-DAG: attributes [[CONVERGENT]] = { convergent }
// CHECK-DAG: attributes [[NOUNWIND]] = { nounwind }
