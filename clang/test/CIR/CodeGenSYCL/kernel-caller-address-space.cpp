// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM-OGCG
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM-OGCG

// SYCL uses the "generic as default address space" deduction mode: unlike
// OpenCL, the address space is not deduced in Sema, so an unqualified pointer
// reaches CodeGen as LangAS::Default and must be given the generic address
// space during device compilation. This verifies that a pointer captured by a
// SYCL kernel is emitted in the generic address space (address space 4 on the
// SPIR-V target), matching classic CodeGen.

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;

void test(int *p) {
  kernel_single_task<KN>([p]() { *p = 42; });
}

// The captured pointer lives in the closure object. Its default address space
// is resolved to the generic address space.
// CIR: !cir.ptr<!s32i, target_address_space(4)>

// The kernel caller entry point receives the closure and casts it to the
// generic address space before invoking the kernel functor.
// CIR-LABEL: cir.func {{.*}}@_ZTS2KN
// CIR:         cir.cast address_space {{.*}} -> !cir.ptr<{{.*}}, target_address_space(4)>

// The kernel functor's operator() reads the captured pointer, which is a
// generic-address-space pointer, and stores through it.
// CIR-LABEL: cir.func {{.*}}@_ZZ4testPiENKUlvE_clEv
// CIR:         cir.get_member {{.*}} -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR:         cir.store {{.*}} : !s32i, !cir.ptr<!s32i, target_address_space(4)>

// The captured pointer field and the store through it use address space 4,
// matching classic CodeGen.
// LLVM-OGCG: %class.anon{{.*}} = type { ptr addrspace(4) }
// LLVM-OGCG-LABEL: define {{.*}}@_ZZ4testPiENKUlvE_clEv(ptr addrspace(4)
// LLVM-OGCG:         store i32 42, ptr addrspace(4)
