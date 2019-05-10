; REQUIRES: x86-registered-target

; Backend test for distribute ThinLTO with CFI.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px

; Check that typeids are in the index.
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s
; CHECK-LABEL: <GLOBALVAL_SUMMARY_BLOCK
; CHECK: <TYPE_ID op0=0 op1=6 op2=3 op3=0 op4=0 op5=0 op6=0 op7=0/>
; CHECK-LABEL: </GLOBALVAL_SUMMARY_BLOCK
; CHECK-LABEL: <STRTAB_BLOCK
; CHECK: blob data = '_ZTS1A'
; CHECK-LABEL: </STRTAB_BLOCK

; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=CHECK-DIS
; Round trip it through llvm-as
; RUN: llvm-dis %t.o.thinlto.bc -o - | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-DIS
; CHECK-DIS: ^0 = module: (path: "{{.*}}thinlto-distributed-cfi.ll.tmp.o", hash: ({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}))
; CHECK-DIS: ^1 = gv: (guid: 8346051122425466633, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 7, typeIdInfo: (typeTests: (^2)))))
; CHECK-DIS: ^2 = typeid: (name: "_ZTS1A", summary: (typeTestRes: (kind: single, sizeM1BitWidth: 0))) ; guid = 7004155349499253778

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -emit-llvm -o - -x ir %t.o | FileCheck %s --check-prefixes=CHECK-IR

; Ensure that backend does not fail generating native code.
; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.B = type { %struct.A }
%struct.A = type { i32 (...)** }

@_ZTV1B = constant { [3 x i8*] } { [3 x i8*] [i8* undef, i8* undef, i8* undef] }, !type !0

; CHECK-IR-LABEL: define void @test
define void @test(i8* %b) {
entry:
  ; Ensure that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap
  %0 = bitcast i8* %b to i8**
  %vtable2 = load i8*, i8** %0
  %1 = tail call i1 @llvm.type.test(i8* %vtable2, metadata !"_ZTS1A")
  br i1 %1, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  ; CHECK-IR-LABEL: ret void
  ret void
}
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(%struct.B* %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
