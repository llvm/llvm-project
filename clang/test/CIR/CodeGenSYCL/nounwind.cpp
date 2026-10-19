// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o - \
// RUN: | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o - \
// RUN: | FileCheck %s -check-prefix=LLVM

// SYCL device code has no exceptions, so every function and call is nounwind.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;

int ext(int);

void test(int *p) {
  kernel_single_task<KN>([p]() { *p = ext(*p); });
}

// CIR: cir.func{{.*}}@_ZTS2KN(
// CIR-SAME: nothrow, nounwind
// CIR: cir.call @_ZZ4testPiENKUlvE_clEv(%{{.*}}) nothrow nounwind
// CIR: cir.func{{.*}}@_ZZ4testPiENKUlvE_clEv(
// CIR-SAME: nothrow, nounwind
// CIR: cir.call @_Z3exti(%{{.*}}) nothrow nounwind
// CIR: cir.func private @_Z3exti(
// CIR-SAME: nothrow, nounwind

// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}@_ZTS2KN(
// LLVM: call {{.*}}@_ZZ4testPiENKUlvE_clEv({{.*}}) #[[CALL_ATTR:[0-9]+]]
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: define {{.*}}@_ZZ4testPiENKUlvE_clEv(
// LLVM: call {{.*}}@_Z3exti({{.*}}) #[[CALL_ATTR]]
// LLVM: ; Function Attrs: {{.*}}nounwind
// LLVM-NEXT: declare {{.*}}@_Z3exti(
// LLVM: attributes #[[CALL_ATTR]] = {{{.*}}nounwind
