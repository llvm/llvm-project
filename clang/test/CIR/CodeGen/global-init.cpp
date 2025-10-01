// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// Note: The LoweringPrepare work isn't yet complete. We still need to create
//       the global ctor list attribute.

struct NeedsCtor {
  NeedsCtor();
};

NeedsCtor needsCtor;

// CIR-BEFORE-LPP: cir.global external @needsCtor = ctor : !rec_NeedsCtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// CIR: cir.global external @needsCtor = #cir.zero : !rec_NeedsCtor
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %0 = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%0) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// CIR: cir.func private @_GLOBAL__sub_I_[[FILENAME:.*]]() {
// CIR:   cir.call @__cxx_global_var_init() : () -> ()
// CIR:   cir.return
// CIR: }
