; REQUIRES: x86-registered-target

; Check that we properly update @llvm.public.type.test with distributed ThinLTO.

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,f,px

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o --save-temps=obj
; RUN: llvm-dis %t.native.o.0.preopt.bc -o - | FileCheck %s --check-prefix=PUBLIC

; RUN: llvm-lto2 run -thinlto-distributed-indexes %t.o --whole-program-visibility \
; RUN:   -o %t2.index \
; RUN:   -r=%t.o,f,px

; RUN: %clang_cc1 -triple x86_64-grtev4-linux-gnu \
; RUN:   -emit-obj -fthinlto-index=%t.o.thinlto.bc \
; RUN:   -o %t.native.o -x ir %t.o --save-temps=obj
; RUN: llvm-dis %t.native.o.0.preopt.bc -o - | FileCheck %s --check-prefix=HIDDEN

; PUBLIC-NOT: call {{.*}}@llvm.public.type.test
; PUBLIC-NOT: call {{.*}}@llvm.type.test
; PUBLIC: call void @llvm.assume(i1 true)

; HIDDEN-NOT: call {{.*}}@llvm.public.type.test
; HIDDEN: call {{.*}}@llvm.type.test

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

define i32 @f(ptr %vtable) {
entry:
  %p = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  ret i32 0
}

declare void @llvm.assume(i1)
declare i1 @llvm.public.type.test(ptr, metadata)
