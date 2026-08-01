; REQUIRES: x86
; RUN: split-file %s %t.dir

; RUN: llvm-as %t.dir/main.ll -o %t.main.bc
; RUN: llvm-as %t.dir/other.ll -o %t.other.bc
; RUN: llc --emulated-tls %t.dir/main.ll -o %t.main.obj --filetype=obj
; RUN: llc --emulated-tls %t.dir/other.ll -o %t.other.obj --filetype=obj
; RUN: llc --emulated-tls %t.dir/runtime.ll -o %t.runtime.obj --filetype=obj

; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.obj %t.other.obj %t.runtime.obj -entry __main -o %t.exe
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.bc  %t.other.bc  %t.runtime.obj -entry __main -o %t.exe
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.bc  %t.other.obj %t.runtime.obj -entry __main -o %t.exe
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.obj %t.other.bc  %t.runtime.obj -entry __main -o %t.exe
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls -shared %t.other.bc %t.runtime.obj -o %t.dll --out-implib=%t.lib
; RUN: llvm-readobj --coff-exports %t.dll | FileCheck %s
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.bc %t.dll %t.runtime.obj -entry __main -o %t.exe
; RUN: llvm-readobj --coff-imports %t.exe | FileCheck %s
; RUN: ld.lld -m i386pep -plugin-opt=mcpu=x86-64 -plugin-opt=-emulated-tls %t.main.bc %t.lib %t.runtime.obj -entry __main -o %t.exe
; RUN: llvm-readobj --coff-imports %t.exe | FileCheck %s

; CHECK: _Z11set_tls_vari
; CHECK: __emutls_v.tls_var

;--- main.ll
;; generated from:
;;;extern int thread_local tls_var;
;;;void set_tls_var(int v);
;;;int main(int argc, char **argv) {
;;;  set_tls_var(3);
;;;  return tls_var == argc;
;;;}
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

$_ZTW7tls_var = comdat any

@tls_var = external dso_local thread_local global i32, align 4

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @__main(i32 noundef %0, ptr noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca ptr, align 8
  store i32 0, ptr %3, align 4
  store i32 %0, ptr %4, align 4
  store ptr %1, ptr %5, align 8
  call void @_Z11set_tls_vari(i32 noundef 3)
  %6 = call ptr @_ZTW7tls_var()
  %7 = load i32, ptr %6, align 4
  %8 = load i32, ptr %4, align 4
  %9 = icmp eq i32 %7, %8
  %10 = zext i1 %9 to i32
  ret i32 %10
}

declare dso_local void @_Z11set_tls_vari(i32 noundef) #1

; Function Attrs: noinline uwtable
define linkonce_odr hidden noundef ptr @_ZTW7tls_var() #2 comdat {
  %1 = icmp ne ptr @_ZTH7tls_var, null
  br i1 %1, label %2, label %3

2:                                                ; preds = %0
  call void @_ZTH7tls_var()
  br label %3

3:                                                ; preds = %2, %0
  %4 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @tls_var)
  ret ptr %4
}

declare extern_weak void @_ZTH7tls_var() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #3

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}

;--- other.ll
;; generated from:
;;;int thread_local tls_var;
;;;void set_tls_var(int v) { tls_var = v; }
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

$_ZTW7tls_var = comdat any

@tls_var = dso_local thread_local global i32 0, align 4

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z11set_tls_vari(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @tls_var)
  store i32 %3, ptr %4, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull) #1

; Function Attrs: noinline uwtable
define weak_odr hidden noundef ptr @_ZTW7tls_var() #2 comdat {
  %1 = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @tls_var)
  ret ptr %1
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ThinLTO", i32 0}

;--- runtime.ll
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

define dso_local hidden noundef i32 @__emutls_get_address() {
  ret i32 0
}

define dso_local hidden noundef i32 @__emutls_register_common() {
  ret i32 0
}

define dso_local hidden noundef i32 @_pei386_runtime_relocator() {
  ret i32 0
}

define dso_local hidden noundef i32 @_DllMainCRTStartup() {
  ret i32 0
}
