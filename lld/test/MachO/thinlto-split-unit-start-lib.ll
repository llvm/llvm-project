; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t

;; Split-LTO bitcode files can have the same symbol (typeinfo) twice in the
;; symbol table: First undefined in the main file, then defined in the index
;; file.
;; When used in --start-lib / --end-lib, ld64.lld creates lazy symbols
;; for all non-undefined symbols in the bitcode file.
;; In vtable.o below, the typeinfo __ZTI1S is present once as undefined and
;; once as defined. The defined version is added as a added as a LazyObject
;; symbol.
;; When vtable.o gets loaded due to the __ZN1SC1Ev ref from vtable_use.o,
;; the first __ZTI1S (undefined) used to cause vtable.o to be extracted
;; a second time, which used to cause an assert.
;; See PR59162 for details.

; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/vtable.o %t/vtable.ll
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t/vtable_use.o %t/vtable_use.ll

; RUN: %lld -lc++ --start-lib %t/vtable.o --end-lib %t/vtable_use.o -o /dev/null

;; Bitcode files created by:
; % cat vtable.cc
; struct S {
;   S();
;   virtual void f();
; };
; S::S() {}
; void S::f() {}

; % cat vtable_use.cc
; struct S {
;   S();
;   virtual void f();
; };
; int main() { S s; }

; % clang -c vtable_use.cc vtable.cc -emit-llvm -S -fno-exceptions -arch x86_64 -mmacos-version-min=11 -O1

; ...and then manually ading `, !type !8, type !9` based on `clang -S -emit-llvm -flto=thin` output,
; because splitAndWriteThinLTOBitcode() in ThinLTOBitcodeWriter.cpp only splits bitcode
; if type annotations are present. While at it, also removed unneccessary metadata.
; (NB: The first comment creates vtable.ll while the latter generates vtable.s! vtable.s
; contains a few things opt complains about, so we can't use the output of that directly.)

;--- vtable.ll
; ModuleID = 'vtable.cc'
source_filename = "vtable.cc"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

@_ZTV1S = unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1S, ptr @_ZN1S1fEv] }, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS1S = constant [3 x i8] c"1S\00", align 1
@_ZTI1S = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1S }, align 8, !type !0, !type !1

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp willreturn memory(argmem: write) uwtable
define void @_ZN1SC2Ev(ptr nocapture noundef nonnull writeonly align 8 dereferenceable(8) %this) unnamed_addr align 2 {
entry:
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1S, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp willreturn memory(argmem: write) uwtable
define void @_ZN1SC1Ev(ptr nocapture noundef nonnull writeonly align 8 dereferenceable(8) %this) unnamed_addr align 2 {
entry:
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1S, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind ssp willreturn memory(none) uwtable
define void @_ZN1S1fEv(ptr nocapture nonnull align 8 %this) unnamed_addr align 2 {
entry:
  ret void
}

!0 = !{i64 16, !"_ZTS1S"}
!1 = !{i64 16, !"_ZTSM1SFvvE.virtual"}

;--- vtable_use.ll
; ModuleID = 'vtable_use.cc'
source_filename = "vtable_use.cc"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

%struct.S = type { ptr }

; Function Attrs: mustprogress noinline norecurse nounwind optnone ssp uwtable
define noundef i32 @main() {
entry:
  %s = alloca %struct.S, align 8
  call void @_ZN1SC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %s)
  ret i32 0
}

declare void @_ZN1SC1Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr
