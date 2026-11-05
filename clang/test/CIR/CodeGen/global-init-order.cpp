// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct NeedsCtor {
  NeedsCtor();
};

struct Holder {
  static NeedsCtor first;
  static NeedsCtor second;
};

NeedsCtor *p = &Holder::first;

NeedsCtor Holder::second;
NeedsCtor Holder::first;

// CIR-BEFORE-LPP: cir.global external @p = #cir.global_view<@_ZN6Holder5firstE> : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP: cir.global external @_ZN6Holder6secondE = #cir.zero : !rec_NeedsCtor ctor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @_ZN6Holder6secondE : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]])
// CIR-BEFORE-LPP: }
// CIR-BEFORE-LPP: cir.global external @_ZN6Holder5firstE = #cir.zero : !rec_NeedsCtor ctor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @_ZN6Holder5firstE : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]])
// CIR-BEFORE-LPP: }

// CIR: cir.global external @p = #cir.global_view<@_ZN6Holder5firstE> : !cir.ptr<!rec_NeedsCtor>
// CIR: cir.global external @_ZN6Holder6secondE = #cir.zero : !rec_NeedsCtor
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %[[SECOND:.*]] = cir.get_global @_ZN6Holder6secondE : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%[[SECOND]])
// CIR: cir.global external @_ZN6Holder5firstE = #cir.zero : !rec_NeedsCtor
// CIR: cir.func internal private @__cxx_global_var_init.1() {
// CIR:   %[[FIRST:.*]] = cir.get_global @_ZN6Holder5firstE : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%[[FIRST]])

// CIR: cir.func internal private @_GLOBAL__sub_I_
// CIR:   cir.call @__cxx_global_var_init() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.1() : () -> ()

// LLVM-DAG: @p = global ptr @_ZN6Holder5firstE, align 8
// LLVM-DAG: @_ZN6Holder6secondE = global %struct.NeedsCtor zeroinitializer, align 1
// LLVM-DAG: @_ZN6Holder5firstE = global %struct.NeedsCtor zeroinitializer, align 1

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @_ZN9NeedsCtorC1Ev(ptr {{.*}} @_ZN6Holder6secondE)

// LLVM: define internal void @__cxx_global_var_init.1()
// LLVM:   call void @_ZN9NeedsCtorC1Ev(ptr {{.*}} @_ZN6Holder5firstE)

// LLVM: define internal void @_GLOBAL__sub_I_
// LLVM:   call void @__cxx_global_var_init()
// LLVM:   call void @__cxx_global_var_init.1()
