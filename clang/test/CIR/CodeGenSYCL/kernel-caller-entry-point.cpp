// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefixes=LLVM,LLVM-OGCG
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=OGCG,LLVM-OGCG

// On an ELF target such as spir64, the kernel caller entry point definition is
// dso_local. dso_local is only attached to a definition, so this also verifies
// that setDSOLocal() runs after body emission.
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-cir %s -o %t-elf.cir
// RUN: FileCheck --input-file=%t-elf.cir %s -check-prefix=CIR-ELF
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown -fclangir -emit-llvm %s -o %t-elf-cir.ll
// RUN: FileCheck --input-file=%t-elf-cir.ll %s -check-prefix=LLVM-OGCG-ELF
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm %s -o %t-elf.ll
// RUN: FileCheck --input-file=%t-elf.ll %s -check-prefix=LLVM-OGCG-ELF

// During device compilation, an offload kernel caller entry point is emitted
// in place of each sycl_kernel_entry_point attributed function. The entry
// point is named after the kernel name type and its body is the transformed
// body held by the OutlinedFunctionDecl (which invokes the kernel functor).
// The sycl_kernel_entry_point attributed function itself is not emitted.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
struct MemberKN;
struct K {
  void operator()() const {}
};

// A sycl_kernel_entry_point function may also be a non-static member function
// (Sema only rejects explicit-object members, ctors and dtors). The offload
// entry point is still a free function and must not run an instance-function
// prologue.
struct Invoker {
  template <typename KernelName, typename KernelType>
  [[clang::sycl_kernel_entry_point(KernelName)]]
  void kernel_single_task(KernelType kf) { kf(); }
};

void test() {
  kernel_single_task<KN>(K{});
  Invoker{}.kernel_single_task<MemberKN>(K{});
}

// The kernel caller entry point is named after the kernel name type (KN), is
// emitted with the spir_kernel calling convention, and its body calls the
// kernel functor's operator(). The sycl_kernel_entry_point function and its
// launch call are not emitted during device compilation.
// CIR-LABEL: cir.func {{.*}}@_ZTS2KN{{.*}}cc(spir_kernel)
// CIR:         cir.call @_ZNK1KclEv
// CIR:         cir.return
// CIR-NOT:   cir.func {{.*}}@_Z18kernel_single_task
// CIR-NOT:   cir.call {{.*}}@_Z17sycl_kernel_launch

// The member-function entry point is emitted the same way, as a free function
// (no implicit `this` parameter).
// CIR-LABEL: cir.func {{.*}}@_ZTS8MemberKN{{.*}}cc(spir_kernel)
// CIR:         cir.call @_ZNK1KclEv
// CIR:         cir.return

// LLVM-OGCG-LABEL: define {{.*}}spir_kernel void @_ZTS2KN
// LLVM:              call {{.*}}void @_ZNK1KclEv
// OGCG:              call {{.*}}spir_func void @_ZNK1KclEv
// LLVM-OGCG:         ret void
// LLVM-OGCG-NOT:   define {{.*}}@_Z18kernel_single_task

// On ELF, the kernel caller entry point definition is dso_local in CIR,
// CIR-lowered LLVM IR, and classic CodeGen alike.
// CIR-ELF:          cir.func {{.*}}dso_local {{.*}}@_ZTS2KN
// LLVM-OGCG-ELF:    define {{.*}}dso_local {{.*}}void @_ZTS2KN
