; REQUIRES: x86-registered-target

; Backend test for distribute ThinLTO with CFI.
; It additionally enables -fwhole-program-vtables to get more information in
; TYPE_IDs of GLOBALVAL_SUMMARY_BLOCK.

; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t.o %s

; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39436.
; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -whole-program-visibility \
; RUN:   -verify-machineinstrs=0 \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,test,px \
; RUN:   -r=%t.o,_ZN1A1nEi,p \
; RUN:   -r=%t.o,_ZN1B1fEi,p \
; RUN:   -r=%t.o,_ZN1C1fEi,p \
; RUN:   -r=%t.o,_ZTV1B, \
; RUN:   -r=%t.o,_ZTV1C, \
; RUN:   -r=%t.o,_ZN1A1nEi, \
; RUN:   -r=%t.o,_ZN1B1fEi, \
; RUN:   -r=%t.o,_ZN1C1fEi, \
; RUN:   -r=%t.o,_ZTV1B,px \
; RUN:   -r=%t.o,_ZTV1C,px

; Ensure that typeids are in the index.
; RUN: llvm-bcanalyzer -dump %t.o.thinlto.bc | FileCheck %s
; CHECK-LABEL: <GLOBALVAL_SUMMARY_BLOCK
; CHECK: <TYPE_ID op0=0 op1=6 op2=4 op3=7 op4=0 op5=0 op6=0 op7=0 op8=0 op9=2 op10=6 op11=0 op12=0 op13=8 op14=1 op15=6 op16=9 op17=0/>
; CHECK-LABEL: </GLOBALVAL_SUMMARY_BLOCK
; CHECK-LABEL: <STRTAB_BLOCK
; CHECK: blob data = '_ZTS1A_ZN1A1nEi'
; CHECK-LABEL: </STRTAB_BLOCK

; RUN: llvm-dis %t.o.thinlto.bc -o - | FileCheck %s --check-prefix=CHECK-DIS
; Round trip it through llvm-as
; RUN: llvm-dis %t.o.thinlto.bc -o - | llvm-as -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-DIS
; CHECK-DIS: ^0 = module: (path: "{{.*}}thinlto-distributed-cfi-devirt.ll.tmp.o", hash: ({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}))
; CHECK-DIS: ^1 = gv: (guid: 8346051122425466633, summaries: (function: (module: ^0, flags: (linkage: external, notEligibleToImport: 0, live: 1, dsoLocal: 0, canAutoHide: 0), insts: 18, typeIdInfo: (typeTests: (^2), typeCheckedLoadVCalls: (vFuncId: (^2, offset: 8), vFuncId: (^2, offset: 0))))))
; CHECK-DIS: ^2 = typeid: (name: "_ZTS1A", summary: (typeTestRes: (kind: allOnes, sizeM1BitWidth: 7), wpdResolutions: ((offset: 0, wpdRes: (kind: branchFunnel)), (offset: 8, wpdRes: (kind: singleImpl, singleImplName: "_ZN1A1nEi"))))) ; guid = 7004155349499253778

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc -O2 \
; RUN:   -emit-llvm -o - -x ir %t.o | FileCheck %s --check-prefixes=CHECK-IR

; Check that backend does not fail generating native code.
; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc -O2 \
; RUN:   -o %t.native.o -x ir %t.o

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }

@_ZTV1B = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1
@_ZTV1C = constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.C*, i32)* @_ZN1C1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !2

; CHECK-IR-LABEL: define i32 @test
define i32 @test(%struct.A* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8**
  %vtable5 = load i8*, i8** %0

  %1 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable5, i32 8, metadata !"_ZTS1A")
  %2 = extractvalue { i8*, i1 } %1, 1
  br i1 %2, label %cont, label %trap

trap:
  tail call void @llvm.trap()
  unreachable

cont:
  %3 = extractvalue { i8*, i1 } %1, 0
  %4 = bitcast i8* %3 to i32 (%struct.A*, i32)*

  ; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %4(%struct.A* nonnull %obj, i32 %a)
  %vtable16 = load i8*, i8** %0
  %5 = tail call { i8*, i1 } @llvm.type.checked.load(i8* %vtable16, i32 0, metadata !"_ZTS1A")
  %6 = extractvalue { i8*, i1 } %5, 1
  br i1 %6, label %cont2, label %trap

cont2:
  %7 = extractvalue { i8*, i1 } %5, 0
  %8 = bitcast i8* %7 to i32 (%struct.A*, i32)*

  ; Check that traps are conditional. Invalid TYPE_ID can cause
  ; unconditional traps.
  ; CHECK-IR: br i1 {{.*}}, label %trap

  ; We still have to call it as virtual.
  ; CHECK-IR: %call3 = tail call i32 %7
  %call3 = tail call i32 %8(%struct.A* nonnull %obj, i32 %call)
  ret i32 %call3
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare { i8*, i1 } @llvm.type.checked.load(i8*, i32, metadata)
declare void @llvm.trap()

declare i32 @_ZN1B1fEi(%struct.B* %this, i32 %a)
declare i32 @_ZN1A1nEi(%struct.A* %this, i32 %a)
declare i32 @_ZN1C1fEi(%struct.C* %this, i32 %a)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 16, !"_ZTS1C"}
