// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Calls in SYCL device code use the spir_func calling convention, matching the
// calling convention of the callee.

template <typename KN, typename F>
[[clang::sycl_kernel_entry_point(KN)]] void kernel(F f) { f(); }

template <typename KN, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

int run2() { return 1; }
void run() { (void)run2(); }

struct KN;
void use() { kernel<KN>([] { run(); }); }

// CIR: cir.func {{.*}}@_ZTS2KN({{.*}}) cc(spir_kernel)
// CIR:   cir.call @_ZZ3usevENKUlvE_clEv(%{{.*}}) cc(spir_function)
// CIR: cir.func {{.*}}@_ZZ3usevENKUlvE_clEv({{.*}}) cc(spir_function)
// CIR:   cir.call @_Z3runv() cc(spir_function)
// CIR: cir.func {{.*}}@_Z3runv() cc(spir_function)
// CIR:   cir.call @_Z4run2v() cc(spir_function)
// CIR: cir.func {{.*}}@_Z4run2v() -> {{.*}} cc(spir_function)

// LLVM-CIR: define dso_local spir_kernel void @_ZTS2KN(
// OGCG:     define dso_local spir_kernel void @_ZTS2KN(
// LLVM:       call spir_func void @_ZZ3usevENKUlvE_clEv(ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %{{.*}})
// LLVM:     define internal spir_func void @_ZZ3usevENKUlvE_clEv(
// LLVM:       call spir_func void @_Z3runv()
// LLVM:     define dso_local spir_func void @_Z3runv()
// LLVM:       call spir_func noundef i32 @_Z4run2v()
// LLVM:     define dso_local spir_func noundef i32 @_Z4run2v()
