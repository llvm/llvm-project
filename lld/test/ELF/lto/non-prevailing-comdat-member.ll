; REQUIRES: x86
; RUN: llvm-as %p/Inputs/non-prevailing-comdat-member-a.ll -o %t.a.bc
; RUN: llvm-as %p/Inputs/non-prevailing-comdat-member-b.ll -o %t.b.bc
; RUN: ld.lld -shared %t.a.bc %t.b.bc -o %t.forward.so
; RUN: llvm-nm -D %t.forward.so | FileCheck %s
; RUN: ld.lld -shared %t.b.bc %t.a.bc -o %t.reverse.so
; RUN: llvm-nm -D %t.reverse.so | FileCheck %s
;
; CHECK: W _ZN1AIiED0Ev
; CHECK-NEXT: W _ZN1AIiED1Ev
; CHECK-NEXT: W _ZN1AIiED2Ev
