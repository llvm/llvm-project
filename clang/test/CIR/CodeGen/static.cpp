// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt %t.cir -o - | FileCheck %s -check-prefix=CIR

class Init {

public:
  Init(bool a) ;

private:
  static bool _S_synced_with_stdio;
};


static Init __ioinit(true);

// CIR:      module {{.*}} {
// CIR-NEXT:   cir.func private @_ZN4InitC1Eb(!cir.ptr<!ty_22class2EInit22>, !cir.bool)
// CIR-NEXT:   cir.global "private" internal @_ZL8__ioinit = ctor : !ty_22class2EInit22 {
// CIR-NEXT:     %0 = cir.get_global @_ZL8__ioinit : cir.ptr <!ty_22class2EInit22>
// CIR-NEXT:     %1 = cir.const(#true) : !cir.bool
// CIR-NEXT:     cir.call @_ZN4InitC1Eb(%0, %1) : (!cir.ptr<!ty_22class2EInit22>, !cir.bool) -> ()
// CIR-NEXT:   }
// CIR-NEXT: }
