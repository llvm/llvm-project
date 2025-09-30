// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// Note: The CIR generated from this test isn't ready for lowering to LLVM yet.
//       That will require changes to LoweringPrepare.

struct NeedsCtor {
  NeedsCtor();
};

NeedsCtor needsCtor;

// CIR: cir.func private @_ZN9NeedsCtorC1Ev(!cir.ptr<!rec_NeedsCtor>)
// CIR: cir.global external @needsCtor = ctor : !rec_NeedsCtor {
// CIR:   %[[THIS:.*]] = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtor>) -> ()
// CIR: }
