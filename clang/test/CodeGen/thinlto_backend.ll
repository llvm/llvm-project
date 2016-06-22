; REQUIRES: x86-registered-target

; RUN: opt -module-summary -o %t1.o %s
; RUN: opt -module-summary -o %t2.o %S/Inputs/thinlto_backend.ll
; RUN: llvm-lto -thinlto -o %t %t1.o %t2.o

; Ensure clang -cc1 give expected error for incorrect input type
; RUN: not %clang_cc1 -O2 -o %t1.o -x c %s -c -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-WARNING
; CHECK-WARNING: error: invalid argument '-fthinlto-index={{.*}}' only allowed with '-x ir'

; Ensure we get expected error for missing index file
; RUN: %clang -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=bad.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
; CHECK-ERROR: Error loading index file 'bad.thinlto.bc'

; Ensure f2 was imported
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc
; RUN: llvm-nm %t3.o | FileCheck --check-prefix=CHECK-OBJ %s
; CHECK-OBJ: T f1
; CHECK-OBJ-NOT: U f2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f2()

define void @f1() {
  call void @f2()
  ret void
}
