; REQUIRES: x86
;; Test that that a vtable defined locally in one module but external in another
;; does not prevent devirtualization.

;; Hybrid WPD
; RUN: split-file %s %t
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/Cat.o %t/Cat.ll
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/User.o %t/User.ll
; RUN: echo '{ global: _Z17useDoThingWithCatv; local: *; };' > %t/version.exp

; RUN: ld.lld %t/Cat.o %t/User.o -shared -o %t/libA.so -save-temps --lto-whole-program-visibility \
; RUN:   -mllvm -pass-remarks=. --version-script %t/version.exp 2>&1 | \
; RUN:   FileCheck %s --check-prefix=REMARK

; REMARK-DAG: <unknown>:0:0: single-impl: devirtualized a call to _ZNK3Cat9makeNoiseEv
; REMARK-DAG: <unknown>:0:0: single-impl: devirtualized a call to _ZNK3Cat9makeNoiseEv

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;--- Cat.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Cat = type { %struct.Animal }
%struct.Animal = type { ptr }

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

@.str = private unnamed_addr constant [5 x i8] c"Meow\00", align 1
@_ZTV3Cat = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI3Cat, ptr @_ZNK3Cat9makeNoiseEv] }, align 8, !type !0, !type !1, !type !2, !type !3
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global ptr
@_ZTS3Cat = dso_local constant [5 x i8] c"3Cat\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global ptr
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS6Animal }, comdat, align 8
@_ZTI3Cat = dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS3Cat, ptr @_ZTI6Animal }, align 8

define dso_local void @_ZNK3Cat9makeNoiseEv(ptr nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr align 2 {
entry:
  %call = tail call i32 @puts(ptr nonnull dereferenceable(1) @.str)
  ret void
}

declare dso_local noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr

define dso_local void @_Z14doThingWithCatP6Animal(ptr %a) local_unnamed_addr {
entry:
  %tobool.not = icmp eq ptr %a, null
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %vtable = load ptr, ptr %a, align 8, !tbaa !4
  %0 = tail call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS3Cat")
  tail call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable, align 8
  tail call void %1(ptr nonnull dereferenceable(8) %a)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)

declare void @llvm.assume(i1 noundef)

!0 = !{i64 16, !"_ZTS3Cat"}
!1 = !{i64 16, !"_ZTSM3CatKFvvE.virtual"}
!2 = !{i64 16, !"_ZTS6Animal"}
!3 = !{i64 16, !"_ZTSM6AnimalKFvvE.virtual"}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}

;--- User.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Animal = type { ptr }
%struct.Cat = type { %struct.Animal }

@_ZTV3Cat = available_externally dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI3Cat, ptr @_ZNK3Cat9makeNoiseEv] }, align 8, !type !0, !type !1, !type !2, !type !3
@_ZTI3Cat = external dso_local constant ptr
@llvm.compiler.used = appending global [1 x ptr] [ptr @_ZTV3Cat], section "llvm.metadata"

declare dso_local void @_ZNK3Cat9makeNoiseEv(ptr nonnull dereferenceable(8)) unnamed_addr

define dso_local void @_Z17useDoThingWithCatv() local_unnamed_addr {
entry:
  %call = tail call noalias nonnull dereferenceable(8) ptr @_Znwm(i64 8)
  store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV3Cat, i64 0, i32 0, i64 2), ptr %call, align 8, !tbaa !4
  tail call void @_Z14doThingWithCatP6Animal(ptr nonnull %call)
  ret void
}

declare dso_local nonnull ptr @_Znwm(i64) local_unnamed_addr

declare dso_local void @_Z14doThingWithCatP6Animal(ptr) local_unnamed_addr

!0 = !{i64 16, !"_ZTS3Cat"}
!1 = !{i64 16, !"_ZTSM3CatKFvvE.virtual"}
!2 = !{i64 16, !"_ZTS6Animal"}
!3 = !{i64 16, !"_ZTSM6AnimalKFvvE.virtual"}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
