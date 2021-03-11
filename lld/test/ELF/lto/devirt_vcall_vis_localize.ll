; REQUIRES: x86
;; If the vtable symbols are localized by a version script, devirtualization
;; can still happen.

; RUN: opt --thinlto-bc -o %t1.o %s
; RUN: opt --thinlto-bc -o %t2.o %S/Inputs/devirt_vcall_vis_shared_def.ll
; RUN: echo '{ global: _start; local: *; };' > %t.ver

; RUN: ld.lld %t1.o %t2.o -o %t.out --save-temps --lto-whole-program-visibility -shared \
; RUN:   -mllvm -pass-remarks=. 2>&1 | count 0

; RUN: ld.lld %t1.o %t2.o -o %t.out --save-temps --lto-whole-program-visibility -shared \
; RUN:   --version-script=%t.ver -mllvm -pass-remarks=. 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis < %t1.o.4.opt.bc | FileCheck %s --check-prefix=CHECK-IR

; REMARK: single-impl: devirtualized a call to _ZN1A1nEi

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }

@_ZTV1A = available_externally unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !vcall_visibility !2
@_ZTV1B = linkonce_odr unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* undef, i8* bitcast (i32 (%struct.B*, i32)* @_ZN1B1fEi to i8*), i8* bitcast (i32 (%struct.A*, i32)* @_ZN1A1nEi to i8*)] }, !type !0, !type !1, !vcall_visibility !2

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [2 x i8*] [ i8* bitcast ( { [4 x i8*] }* @_ZTV1A to i8*), i8* bitcast ( { [4 x i8*] }* @_ZTV1B to i8*)]

; CHECK-IR-LABEL: @_start(
define i32 @_start(%struct.A* %obj, i32 %a) {
entry:
  %0 = bitcast %struct.A* %obj to i8***
  %vtable = load i8**, i8*** %0
  %1 = bitcast i8** %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %1, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr i8*, i8** %vtable, i32 1
  %2 = bitcast i8** %fptrptr to i32 (%struct.A*, i32)**
  %fptr1 = load i32 (%struct.A*, i32)*, i32 (%struct.A*, i32)** %2, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  %call = tail call i32 %fptr1(%struct.A* nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

define available_externally i32 @_ZN1A1fEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0
}

define available_externally i32 @_ZN1A1nEi(%struct.A* %this, i32 %a) #0 {
   ret i32 0
}

define linkonce_odr i32 @_ZN1B1fEi(%struct.B* %this, i32 %a) #0 {
   ret i32 0
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 0}
