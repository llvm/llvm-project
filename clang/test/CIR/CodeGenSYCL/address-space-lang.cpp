// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-cir \
// RUN:   -mmlir --mlir-print-ir-before=cir-target-lowering \
// RUN:   %s -o %t.cir 2> %t.pre.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.pre.cir
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown \
// RUN:   -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-host -triple x86_64-linux-gnu \
// RUN:   -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-host-cir.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t-host-cir.ll
// RUN: %clang_cc1 -std=c++20 -fsycl-is-host -triple x86_64-linux-gnu \
// RUN:   -Wno-deprecated-attributes -emit-llvm %s -o %t-host.ll
// RUN: FileCheck %s --check-prefix=HOST --input-file=%t-host.ll

// SYCL address spaces map onto the same CIR language address spaces as their
// OpenCL counterparts, and are lowered to the same target address spaces as
// in classic CodeGen. On the host, they only affect mangling.

void foo(int [[clang::sycl_global]] *);
void foo(int [[clang::sycl_local]] *);
void foo(int [[clang::sycl_private]] *);
void foo(int [[clang::sycl_generic]] *);
void foo(int [[clang::sycl_constant]] *);
// sycl_global_device and sycl_global_host are only spelled via the OpenCL
// attributes in SYCL mode.
void foo(int [[clang::opencl_global_device]] *);
void foo(int [[clang::opencl_global_host]] *);

// Required by sycl_kernel_entry_point semantics.
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;

void test() {
  kernel_single_task<KN>([]() {
    int [[clang::sycl_global]] *glob;
    int [[clang::sycl_local]] *loc;
    int [[clang::sycl_private]] *priv;
    int [[clang::sycl_generic]] *gen;
    int [[clang::sycl_constant]] *cnst;
    int [[clang::opencl_global_device]] *dev;
    int [[clang::opencl_global_host]] *host;
    foo(glob);
    foo(loc);
    foo(priv);
    foo(gen);
    foo(cnst);
    foo(dev);
    foo(host);
  });
}

#ifndef __SYCL_DEVICE_ONLY__
// The kernel body is not emitted on the host, so reference the declarations
// from a host function.
void host_test(int [[clang::sycl_global]] *glob,
               int [[clang::sycl_local]] *loc,
               int [[clang::sycl_private]] *priv,
               int [[clang::sycl_generic]] *gen,
               int [[clang::sycl_constant]] *cnst,
               int [[clang::opencl_global_device]] *dev,
               int [[clang::opencl_global_host]] *host) {
  foo(glob);
  foo(loc);
  foo(priv);
  foo(gen);
  foo(cnst);
  foo(dev);
  foo(host);
}
#endif

// CIR: cir.func {{.*}} @_Z3fooPU3AS1i(!cir.ptr<!s32i, lang_address_space(offload_global)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS3i(!cir.ptr<!s32i, lang_address_space(offload_local)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS0i(!cir.ptr<!s32i, lang_address_space(offload_private)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS4i(!cir.ptr<!s32i, lang_address_space(offload_generic)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS2i(!cir.ptr<!s32i, lang_address_space(offload_constant)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS5i(!cir.ptr<!s32i, lang_address_space(offload_global_device)>
// CIR: cir.func {{.*}} @_Z3fooPU3AS6i(!cir.ptr<!s32i, lang_address_space(offload_global_host)>

// LLVM-DAG: declare spir_func void @_Z3fooPU3AS1i(ptr addrspace(1) noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS0i(ptr noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS4i(ptr addrspace(4) noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS2i(ptr addrspace(2) noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS5i(ptr addrspace(5) noundef)
// LLVM-DAG: declare spir_func void @_Z3fooPU3AS6i(ptr addrspace(6) noundef)

// HOST-DAG: declare void @_Z3fooPU8SYglobali(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU7SYlocali(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU9SYprivatei(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU9SYgenerici(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU10SYconstanti(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU8SYdevicei(ptr noundef)
// HOST-DAG: declare void @_Z3fooPU6SYhosti(ptr noundef)
