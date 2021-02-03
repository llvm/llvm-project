; REQUIRES: x86
; RUN: rm -rf %t; split-file %s %t
; RUN: llvm-as %t/foo.ll -o %t/foo.o
; RUN: llvm-as %t/objc.ll -o %t/objc.o
; RUN: llvm-ar rcs %t/foo.a %t/foo.o
; RUN: llvm-ar rcs %t/objc.a %t/objc.o

; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
; RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/references-foo.s -o %t/references-foo.o

; RUN: %lld -lSystem %t/references-foo.o %t/foo.a -o %t/test
; RUN: llvm-objdump --macho --syms %t/test | FileCheck %s --check-prefix=FOO

; RUN: %lld -lSystem -force_load %t/foo.a %t/main.o -o %t/force-load
; RUN: llvm-objdump --macho --syms %t/force-load | FileCheck %s --check-prefix=FOO

; RUN: %lld -lSystem %t/foo.a %t/main.o -o %t/no-force-load
; RUN: llvm-objdump --macho --syms %t/no-force-load | FileCheck %s --check-prefix=NO-FOO

; RUN: %lld -lSystem -ObjC -framework CoreFoundation %t/objc.a %t/main.o -o %t/objc
; RUN: llvm-objdump --macho --syms %t/objc | FileCheck %s --check-prefix=OBJC

; RUN: %lld -lSystem -framework CoreFoundation %t/objc.a %t/main.o -o %t/no-objc
; RUN: llvm-objdump --macho --syms %t/no-objc | FileCheck %s --check-prefix=NO-OBJC

; FOO: _foo

; NO-FOO-NOT: _foo

; OBJC-DAG: _OBJC_CLASS_$_Foo
; OBJC-DAG: _OBJC_METACLASS_$_Foo

; NO-OBJC-NOT: _OBJC_CLASS_$_Foo
; NO-OBJC-NOT: _OBJC_METACLASS_$_Foo

;--- foo.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

;--- objc.ll

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

%struct._class_t = type { %struct._class_t*, %struct._class_t*, %struct._class_ro_t* }
%struct._class_ro_t = type { i8* }

@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@OBJC_CLASS_NAME_ = private unnamed_addr constant [4 x i8] c"Foo\00",
  section "__TEXT,__objc_classname,cstring_literals"
@"_OBJC_METACLASS_RO_$_Foo" = internal global %struct._class_ro_t {
    i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0)
  },
  section "__DATA, __objc_const"
@"OBJC_METACLASS_$_Foo" = global %struct._class_t {
    %struct._class_t* @"OBJC_METACLASS_$_NSObject",
    %struct._class_t* @"OBJC_METACLASS_$_NSObject",
    %struct._class_ro_t* @"_OBJC_METACLASS_RO_$_Foo"
  },
  section "__DATA, __objc_data"
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"_OBJC_CLASS_RO_$_Foo" = internal global %struct._class_ro_t {
    i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_CLASS_NAME_, i32 0, i32 0)
  },
  section "__DATA, __objc_const"
@"OBJC_CLASS_$_Foo" = global %struct._class_t {
    %struct._class_t* @"OBJC_METACLASS_$_Foo",
    %struct._class_t* @"OBJC_CLASS_$_NSObject",
    %struct._class_ro_t* @"_OBJC_CLASS_RO_$_Foo"
  },
  section "__DATA, __objc_data"

;--- main.s

.globl _main
_main:
  ret

;--- references-foo.s

.globl _main
_main:
  callq _foo
  ret
