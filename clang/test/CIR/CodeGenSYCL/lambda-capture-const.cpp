// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM-OGCG
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-OGCG

// Verify that references to constant-initialized const variables captured by
// a SYCL kernel lambda are emitted as constants in device code, matching
// classic codegen.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN1;

void capture_const_float(float *out) {
  const float k = 0.9f;
  kernel_single_task<KN1>([=]() { *out = k; });
}

// CIR-LABEL: cir.func {{.*}} @_ZZ19capture_const_floatPfENKUlvE_clEv(
// CIR-NOT:     cir.get_member %{{.*}}[1]
// CIR:         %[[K:.*]] = cir.const #cir.fp<0.899999976> : !cir.float
// CIR:         cir.store {{.*}}%[[K]]

// LLVM-OGCG-LABEL: define internal spir_func void @_ZZ19capture_const_floatPfENKUlvE_clEv(
// LLVM-OGCG-NOT:     getelementptr {{.*}}, i32 0, i32 1
// LLVM-OGCG:         store float f0x3F666666
