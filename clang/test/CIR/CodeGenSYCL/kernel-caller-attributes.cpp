// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm -fno-finite-loops %s -o %t-nomp.ll
// RUN: FileCheck --input-file=%t-nomp.ll %s -check-prefix=LLVM-NOMP

// The SYCL kernel caller offload entry point receives the SYCL 2020 device
// language attributes: it must not recurse (norecurse) and is guaranteed to
// make forward progress in C++11 and later (mustprogress). It also carries the
// "sycl-module-id" attribute, marking it as an entry point for
// per-translation-unit device-code splitting. This matches classic CodeGen's
// SetSYCLKernelAttributes and addSYCLModuleIdAttr.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;

void test(int *p) {
  kernel_single_task<KN>([p]() { *p = 42; });
}

// CIR-LABEL: cir.func
// CIR-SAME:    @_ZTS2KN
// CIR-SAME:    cc(spir_kernel)
// CIR-SAME:    mustprogress
// CIR-SAME:    norecurse
// CIR-SAME:    "sycl-module-id" = "{{.*}}kernel-caller-attributes.cpp"

// LLVM: define spir_kernel void @_ZTS2KN({{.*}}) #[[KATTR:[0-9]+]]
// LLVM: attributes #[[KATTR]] = {{[{].*}}mustprogress{{.*}}norecurse{{.*}}"sycl-module-id"="{{.*}}kernel-caller-attributes.cpp"

// OGCG: define spir_kernel void @_ZTS2KN({{.*}}) #[[KATTR:[0-9]+]]
// OGCG: attributes #[[KATTR]] = {{[{].*}}mustprogress{{.*}}norecurse{{.*}}"sycl-module-id"="{{.*}}kernel-caller-attributes.cpp"

// With -fno-finite-loops, checkIfFunctionMustProgress() is false, so the kernel
// caller must not carry mustprogress; norecurse and sycl-module-id remain. The
// exact attribute set (which omits mustprogress) is verified here.
// LLVM-NOMP: define spir_kernel void @_ZTS2KN({{.*}}) #[[KATTR:[0-9]+]]
// LLVM-NOMP: attributes #[[KATTR]] = { convergent noinline norecurse "sycl-module-id"="{{.*}}kernel-caller-attributes.cpp" }
