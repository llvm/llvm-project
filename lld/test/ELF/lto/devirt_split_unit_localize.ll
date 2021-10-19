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
%struct.Animal = type { i32 (...)** }

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

@.str = private unnamed_addr constant [5 x i8] c"Meow\00", align 1
@_ZTV3Cat = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI3Cat to i8*), i8* bitcast (void (%struct.Cat*)* @_ZNK3Cat9makeNoiseEv to i8*)] }, align 8, !type !0, !type !1, !type !2, !type !3
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global i8*
@_ZTS3Cat = dso_local constant [5 x i8] c"3Cat\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6Animal, i32 0, i32 0) }, comdat, align 8
@_ZTI3Cat = dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Cat, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, align 8

define dso_local void @_ZNK3Cat9makeNoiseEv(%struct.Cat* nocapture nonnull readnone dereferenceable(8) %this) unnamed_addr align 2 {
entry:
  %call = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0))
  ret void
}

declare dso_local noundef i32 @puts(i8* nocapture noundef readonly) local_unnamed_addr

define dso_local void @_Z14doThingWithCatP6Animal(%struct.Animal* %a) local_unnamed_addr {
entry:
  %tobool.not = icmp eq %struct.Animal* %a, null
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %struct.Animal* %a to %struct.Cat*
  %1 = bitcast %struct.Animal* %a to void (%struct.Cat*)***
  %vtable = load void (%struct.Cat*)**, void (%struct.Cat*)*** %1, align 8, !tbaa !4
  %2 = bitcast void (%struct.Cat*)** %vtable to i8*
  %3 = tail call i1 @llvm.type.test(i8* %2, metadata !"_ZTS3Cat")
  tail call void @llvm.assume(i1 %3)
  %4 = load void (%struct.Cat*)*, void (%struct.Cat*)** %vtable, align 8
  tail call void %4(%struct.Cat* nonnull dereferenceable(8) %0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)

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

%struct.Animal = type { i32 (...)** }
%struct.Cat = type { %struct.Animal }

@_ZTV3Cat = available_externally dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast (i8** @_ZTI3Cat to i8*), i8* bitcast (void (%struct.Cat*)* @_ZNK3Cat9makeNoiseEv to i8*)] }, align 8, !type !0, !type !1, !type !2, !type !3
@_ZTI3Cat = external dso_local constant i8*
@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast ({ [3 x i8*] }* @_ZTV3Cat to i8*)], section "llvm.metadata"

declare dso_local void @_ZNK3Cat9makeNoiseEv(%struct.Cat* nonnull dereferenceable(8)) unnamed_addr

define dso_local void @_Z17useDoThingWithCatv() local_unnamed_addr {
entry:
  %call = tail call noalias nonnull dereferenceable(8) i8* @_Znwm(i64 8)
  %0 = bitcast i8* %call to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !4
  %1 = bitcast i8* %call to %struct.Animal*
  tail call void @_Z14doThingWithCatP6Animal(%struct.Animal* nonnull %1)
  ret void
}

declare dso_local nonnull i8* @_Znwm(i64) local_unnamed_addr

declare dso_local void @_Z14doThingWithCatP6Animal(%struct.Animal*) local_unnamed_addr

!0 = !{i64 16, !"_ZTS3Cat"}
!1 = !{i64 16, !"_ZTSM3CatKFvvE.virtual"}
!2 = !{i64 16, !"_ZTS6Animal"}
!3 = !{i64 16, !"_ZTSM6AnimalKFvvE.virtual"}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
