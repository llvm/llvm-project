; REQUIRES: x86

;; Common artifacts
; RUN: opt --thinlto-bc -o %t1.o %s
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t1_hybrid.o %s
; RUN: cp %s %t1_regular.ll
; RUN: echo '!llvm.module.flags = !{!2, !3}' >> %t1_regular.ll
; RUN: echo '!2 = !{i32 1, !"ThinLTO", i32 0}' >> %t1_regular.ll
; RUN: echo '!3 = !{i32 1, !"EnableSplitLTOUnit", i32 1}' >> %t1_regular.ll
; RUN: opt -module-summary -o %t1_regular.o %t1_regular.ll

; RUN: llvm-as %S/Inputs/devirt_validate_vtable_typeinfos_ref.ll -o %t2.bc
; RUN: llc -relocation-model=pic -filetype=obj %t2.bc -o %t2.o

;; Native objects can contain only a reference to the base type infos if the base declaration has no key functions.
;; Because of that, --lto-validate-all-vtables-have-type-infos needs to query for the type info symbol inside native files rather than the
;; type name symbol that's used as the key in !type metadata to correctly stop devirtualization on the native type.

;; Index based WPD
; RUN: ld.lld %t1.o %t2.o -o %t3_index -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s
; RUN: llvm-dis %t1.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-IR

;; Hybrid WPD
; RUN: ld.lld %t1_hybrid.o %t2.o -o %t3_hybrid -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s
; RUN: llvm-dis %t1_hybrid.o.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-IR

;; Regular LTO WPD
; RUN: ld.lld %t1_regular.o %t2.o -o %t3_regular -save-temps --lto-whole-program-visibility --lto-validate-all-vtables-have-type-infos \
; RUN:   -mllvm -pass-remarks=. 2>&1 | FileCheck %s
; RUN: llvm-dis %t3_regular.0.4.opt.bc -o - | FileCheck %s --check-prefixes=CHECK-IR

; CHECK-NOT:     single-impl: devirtualized a call to _ZN1A3fooEv

;; Source code:
;; cat > a.h <<'eof'
;; struct A { virtual int foo(); };
;; int bar(A *a);
;; eof
;; cat > main.cc <<'eof'
;; #include "a.h"
;;
;; int A::foo() { return 1; }
;; int bar(A *a) { return a->foo(); }
;;
;; extern int baz();
;; int main() {
;;   A a;
;;   int i = bar(&a);
;;   int j = baz();
;;   return i + j;
;; }
;; eof
;; clang++ -fwhole-program-vtables -fno-split-lto-unit -flto=thin main.cc -c

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { %struct.Abase }
%struct.Abase = type { ptr }

@_ZTV1A = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1
@_ZTS1A = dso_local constant [3 x i8] c"1A\00", align 1
@_ZTI1A = dso_local constant { ptr, ptr } { ptr null, ptr @_ZTS1A }, align 8

define dso_local noundef i32 @_ZN1A3fooEv(ptr noundef nonnull align 8 dereferenceable(8) %this) #0 align 2 {
entry:
  %this.addr = alloca ptr
  store ptr %this, ptr %this.addr
  %this1 = load ptr, ptr %this.addr
  ret i32 1
}

; CHECK-IR: define dso_local noundef i32 @_Z3barP1A
define dso_local noundef i32 @_Z3barP1A(ptr noundef %a) #0 {
entry:
  %a.addr = alloca ptr
  store ptr %a, ptr %a.addr
  %0 = load ptr, ptr %a.addr
  %vtable = load ptr, ptr %0
  %1 = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %1)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %fptr = load ptr, ptr %vfn
  ;; Check that the call was not devirtualized.
  ; CHECK-IR: %call = call noundef i32 %fptr
  %call = call noundef i32 %fptr(ptr noundef nonnull align 8 dereferenceable(8) %0)
  ret i32 %call
}
; CHECK-IR: ret i32
; CHECK-IR: }

declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1 noundef)

define dso_local noundef i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %struct.A, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %a)
  %call = call noundef i32 @_Z3barP1A(ptr noundef %a)
  store i32 %call, ptr %i, align 4
  %call1 = call noundef i32 @_Z3bazv()
  store i32 %call1, ptr %j, align 4
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %j, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

define linkonce_odr dso_local void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) #0 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1A, i32 0, inrange i32 0, i32 2), ptr %this1, align 8
  ret void
}

declare noundef i32 @_Z3bazv()

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFivE.virtual"}
