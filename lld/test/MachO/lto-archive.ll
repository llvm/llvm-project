; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-as %t/has-objc-symbol.ll -o %t/has-objc-symbol.o
; RUN: llvm-as %t/has-objc-category.ll -o %t/has-objc-category.o
; RUN: llvm-ar rcs %t/foo.a %t/foo.o
; RUN: llvm-ar rcs %t/objc.a %t/has-objc-symbol.o %t/has-objc-category.o

; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/references-foo.s -o %t/references-foo.o

; RUN: %lld -lSystem %t/references-foo.o %t/foo.a -o /dev/null -why_load | FileCheck %s --check-prefix=FOO
; FOO: _foo forced load of foo.o

; RUN: %lld -lSystem -force_load %t/foo.a %t/main.o -o /dev/null -why_load | FileCheck %s --check-prefix=FORCE-LOAD
; FORCE-LOAD: -force_load forced load of foo.o

; RUN: %lld -lSystem -ObjC -framework CoreFoundation %t/objc.a %t/main.o \
; RUN:   -o /dev/null -why_load | FileCheck %s --check-prefix=OBJC
; OBJC: -ObjC forced load of has-objc-category.o
; OBJC: _OBJC_CLASS_$_Foo forced load of has-objc-symbol.o

; RUN: %lld -lSystem %t/foo.a %t/main.o -o %t/no-force-load -why_load | \
; RUN:   FileCheck %s --allow-empty --check-prefix=NO-LOAD

; RUN: %lld -lSystem -framework CoreFoundation %t/objc.a %t/main.o -o %t/no-objc -why_load | \
; RUN:   FileCheck %s --allow-empty --check-prefix=NO-LOAD

; NO-LOAD-NOT: forced load

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- has-objc-symbol.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct._class_t = type { i8 }
@"OBJC_CLASS_$_Foo" = global %struct._class_t { i8 123 }

;--- has-objc-category.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct._category_t = type { i8 }

@"_OBJC_$_CATEGORY_Foo_$_Bar" = internal global %struct._category_t { i8 123 },
  section "__DATA, __objc_const", align 8

@"OBJC_LABEL_CATEGORY_$" = private global [1 x i8*] [
  i8* bitcast (%struct._category_t* @"_OBJC_$_CATEGORY_Foo_$_Bar" to i8*)
  ], section "__DATA,__objc_catlist,regular,no_dead_strip", align 8

@llvm.compiler.used = appending global [1 x i8*] [
  i8* bitcast ([1 x i8*]* @"OBJC_LABEL_CATEGORY_$" to i8*)
  ], section "llvm.metadata"

;--- main.s

.globl _main
_main:
  ret

;--- references-foo.s

.globl _main
_main:
  callq _foo
  ret
