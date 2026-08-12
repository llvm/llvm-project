// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports parameters of an empty or tag class.
// RUN: %clang_cc1 -std=c++20 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-host -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -fsycl-is-host -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Verify that, during host compilation, the body of a function declared with
// the sycl_kernel_entry_point attribute is lowered to its kernel launch
// statement rather than reporting a not-yet-implemented error. The kernel
// entry point body must be replaced by the launch call; the original kernel
// functor invocation must not be emitted on the host.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
struct K {
  void operator()() const {}
};

void test() { kernel_single_task<KN>(K{}); }

// The kernel entry point body is replaced by a call to the sycl_kernel_launch
// specialization, and does not invoke the kernel functor's operator() on the
// host.
// CIR-LABEL: cir.func {{.*}}@_Z18kernel_single_taskI2KN1KEvT0_
// CIR-NOT:     cir.call @_ZNK1KclEv
// CIR:         cir.call @_Z18sycl_kernel_launchI2KNJ1KEEvPKcDpT0_
// CIR:         cir.return

// LLVM-LABEL: define {{.*}}void @_Z18kernel_single_taskI2KN1KEvT0_
// LLVM-NOT:     call {{.*}}@_ZNK1KclEv
// LLVM:         call void @_Z18sycl_kernel_launchI2KNJ1KEEvPKcDpT0_
// LLVM:         ret void

// OGCG-LABEL: define {{.*}}void @_Z18kernel_single_taskI2KN1KEvT0_
// OGCG-NOT:     call {{.*}}@_ZNK1KclEv
// OGCG:         call void @_Z18sycl_kernel_launchI2KNJ1KEEvPKcDpT0_
// OGCG:         ret void
