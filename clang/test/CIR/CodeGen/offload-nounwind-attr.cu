// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM

// Device code cannot unwind even when exceptions are enabled, so calls inside
// an EH cleanup scope must stay plain calls.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fcxx-exceptions -fexceptions -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM

struct D {
  __attribute__((device)) ~D();
};

extern "C" {
__attribute__((device)) int ext(int);

__attribute__((device)) int caller(int x) {
  D d;
  return ext(x);
}
}

// CIR: cir.func{{.*}}@caller(
// CIR-SAME: nothrow, nounwind
// CIR-NOT: cir.try_call
// CIR: cir.call @ext(%{{.*}}) nothrow nounwind
// CIR-NOT: cir.try_call
// CIR: cir.call @_ZN1DD1Ev(%{{.*}}) nothrow nounwind
// CIR-NOT: cir.try_call
// CIR: cir.func private @ext(
// CIR-SAME: nothrow, nounwind

// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}@caller({{.*}}){{.*}} #{{[0-9]+}} {
// LLVM-NOT: invoke
// LLVM: call {{.*}}@ext({{.*}}) #[[CALL_ATTR:[0-9]+]]
// LLVM-NOT: invoke
// LLVM: call void @_ZN1DD1Ev({{.*}}) #[[CALL_ATTR]]
// LLVM-NOT: landingpad
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: declare {{.*}}@ext(
// LLVM: attributes #[[CALL_ATTR]] = {{{.*}}nounwind
