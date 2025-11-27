; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

; RUN: %lld -dylib --separate-cstring-literal-sections %t.o -o - | llvm-objdump --macho --section-headers - | FileCheck %s
; RUN: %lld -dylib --no-separate-cstring-literal-sections %t.o -o - | llvm-objdump --macho --section-headers - | FileCheck %s --check-prefix=CSTR
; RUN: %lld -dylib %t.o -o - | llvm-objdump --macho --section-headers - | FileCheck %s --check-prefix=CSTR

; CHECK-DAG: __cstring
; CHECK-DAG: __new_sec
; CHECK-DAG: __objc_classname
; CHECK-DAG: __objc_methname
; CHECK-DAG: __objc_methtype

; CSTR-DAG: __cstring
; CSTR-DAG: __objc_methname

target triple = "x86_64-apple-darwin"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"

@.str = private unnamed_addr constant [10 x i8] c"my string\00", align 1
@.str1 = private unnamed_addr constant [16 x i8] c"my other string\00", section "__TEXT,__new_sec,cstring_literals", align 1
@OBJC_CLASS_NAME_ = private unnamed_addr constant [4 x i8] c"foo\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [4 x i8] c"bar\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [4 x i8] c"goo\00", section "__TEXT,__objc_methtype,cstring_literals", align 1

@llvm.compiler.used = appending global [5 x ptr] [
  ptr @.str,
  ptr @.str1,
  ptr @OBJC_METH_VAR_NAME_,
  ptr @OBJC_CLASS_NAME_,
  ptr @OBJC_METH_VAR_TYPE_
]
