// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// Verify that the "sycl-module-id" function attribute is emitted for:
// 1. SYCL kernels
// 2. sycl_external functions

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
struct K { void operator()() const {} };
void use() { kernel_single_task<KN>(K{}); }

[[clang::sycl_external]] int ext(int x) { return x + 1; }

// Both the kernel and the sycl_external function must carry sycl-module-id
// with a value equal to the module identifier (the source file path).
// CIR: module @"[[MODID:.*]]" attributes
// CIR-DAG: cir.func {{.*}}@_Z3exti({{.*}} attributes {{{.*}}"sycl-module-id" = "[[MODID]]"}
// CIR-DAG: cir.func {{.*}}@_ZTS2KN({{.*}} attributes {{{.*}}"sycl-module-id" = "[[MODID]]"}

// LLVM: source_filename = "[[MODID:.*]]"
// LLVM-DAG: define {{.*}}spir_kernel {{.*}}@{{.*}}2KN{{.*}} #[[SATTR:[0-9]+]]
// LLVM-DAG: define {{.*}}spir_func {{.*}}@_Z3exti{{.*}} #[[SATTR]]
// LLVM: attributes #[[SATTR]] = { {{.*}}"sycl-module-id"="[[MODID]]"
