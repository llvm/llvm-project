; REQUIRES: x86

; Check that we RAUW llvm.public.type.test with either llvm.type.test when --lto-whole-program-visibility is specified, or with true otherwise.

; RUN: opt --thinlto-bc -o %t.o %s
; RUN: ld.lld %t.o -o %t2.o --save-temps
; RUN: llvm-dis %t.o.0.preopt.bc -o - | FileCheck %s --check-prefix=PUB
; RUN: ld.lld %t.o -o %t3.o --save-temps --lto-whole-program-visibility
; RUN: llvm-dis %t.o.0.preopt.bc -o - | FileCheck %s --check-prefix=WPV

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1)

; PUB-NOT: call {{.*}}@llvm.public.type.test
; PUB-NOT: call {{.*}}@llvm.type.test
; PUB: call {{.*}}@llvm.assume(i1 true)
; WPV: call {{.*}}@llvm.type.test
; WPV: call {{.*}}@llvm.assume

define void @f(ptr %a) {
  %i = call i1 @llvm.public.type.test(ptr %a, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %i)
  ret void
}
