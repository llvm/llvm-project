// RUN: %clang_cc1 %s -fclangir -emit-cir -triple spirv64-unknown-unknown -o - \
// RUN: | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 %s -fclangir -emit-llvm -triple spirv64-unknown-unknown -o - \
// RUN: | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 %s -emit-llvm -triple spirv64-unknown-unknown -o - \
// RUN: | FileCheck %s --check-prefix=LLVM

// OpenCL has no exceptions, so every function and call is nounwind.

int ext(int);

int caller(int x) { return ext(x); }

// CIR: cir.func{{.*}}@caller(
// CIR-SAME: nothrow, nounwind
// CIR: cir.call @ext(%{{.*}}) nothrow nounwind
// CIR: cir.func private @ext(
// CIR-SAME: nothrow, nounwind

// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}@caller(
// LLVM: call {{.*}}@ext({{.*}}) #[[CALL_ATTR:[0-9]+]]
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: declare {{.*}}@ext(
// LLVM: attributes #[[CALL_ATTR]] = {{{.*}}nounwind
