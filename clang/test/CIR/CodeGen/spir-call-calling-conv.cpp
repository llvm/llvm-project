// RUN: %clang_cc1 -triple spir64 -disable-llvm-passes -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple spir64 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Calls that ClangIR builds directly to C++ functions on SPIR use the callee's
// spir_func calling convention.

struct S { S(); ~S(); int x; };

// Global variable and reference temporary destructors.
S g;
const S &r = S();

// CIR-BEFORE-LPP: cir.global external @g = ctor : !rec_S {
// CIR-BEFORE-LPP:   cir.call @_ZN1SC1Ev(%{{.*}}) cc(spir_function)
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   cir.call @_ZN1SD1Ev(%{{.*}}) cc(spir_function)
// CIR-BEFORE-LPP: cir.global external @r = ctor : !cir.ptr<!rec_S> {
// CIR-BEFORE-LPP:   cir.call @_ZN1SC1Ev(%{{.*}}) cc(spir_function)
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   cir.call @_ZN1SD1Ev(%{{.*}}) cc(spir_function)

// TODO(cir): The global init functions and the __cxa_atexit call do not use
// the runtime calling convention yet.
// LLVM-CIR: define internal void @__cxx_global_var_init()
// OGCG:     define internal spir_func void @__cxx_global_var_init()
// LLVM:       call spir_func void @_ZN1SC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @g)
// LLVM-CIR:   call i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @g, ptr @__dso_handle)
// OGCG:       call spir_func i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @g, ptr @__dso_handle)
// LLVM-CIR: define internal void @__cxx_global_var_init.1()
// OGCG:     define internal spir_func void @__cxx_global_var_init.1()
// LLVM:       call spir_func void @_ZN1SC1Ev(ptr noundef nonnull align 4 dereferenceable(4) @_ZGR1r_)
// LLVM-CIR:   call i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @_ZGR1r_, ptr @__dso_handle)
// OGCG:       call spir_func i32 @__cxa_atexit(ptr @_ZN1SD1Ev, ptr @_ZGR1r_, ptr @__dso_handle)

// Array delete: the element destructor and operator delete[].
void del(S *p) { delete[] p; }

// CIR: cir.func {{.*}}@_Z3delP1S({{.*}}) cc(spir_function)
// CIR:   cir.call @_ZN1SD1Ev(%{{.*}}) cc(spir_function) nothrow
// CIR:   cir.call @_ZdaPvm(%{{.*}}, %{{.*}}) cc(spir_function) nothrow

// LLVM: define dso_local spir_func void @_Z3delP1S(
// LLVM:   call spir_func void @_ZN1SD1Ev(ptr {{.*}}%{{.*}})
// LLVM:   call spir_func void @_ZdaPvm(ptr {{.*}}%{{.*}}, i64 {{.*}}%{{.*}})

// This-adjusting thunk.
struct A { virtual void f(); int a; };
struct D { virtual void g(); int d; };
struct E : A, D { void g() override; };
void E::g() {}

// CIR: cir.func {{.*}}@_ZThn16_N1E1gEv({{.*}}) cc(spir_function)
// CIR:   cir.call @_ZN1E1gEv(%{{.*}}) cc(spir_function)

// LLVM: define dso_local spir_func void @_ZThn16_N1E1gEv(
// LLVM:   call spir_func void @_ZN1E1gEv(ptr noundef nonnull align 8 dereferenceable(28) %{{.*}})

// LLVM-CIR: define internal void @_GLOBAL__sub_I_spir_call_calling_conv.cpp()
// OGCG:     define internal spir_func void @_GLOBAL__sub_I_spir_call_calling_conv.cpp()
// LLVM-CIR:   call void @__cxx_global_var_init()
// OGCG:       call spir_func void @__cxx_global_var_init()
