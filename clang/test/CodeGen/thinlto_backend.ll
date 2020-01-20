; REQUIRES: x86-registered-target

; RUN: opt -module-summary -o %t1.o %s
; RUN: opt -module-summary -o %t2.o %S/Inputs/thinlto_backend.ll
; RUN: llvm-lto -thinlto -o %t %t1.o %t2.o

; Ensure clang -cc1 give expected error for incorrect input type
; RUN: not %clang_cc1 -O2 -o %t1.o -x c %s -c -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-WARNING
; CHECK-WARNING: error: invalid argument '-fthinlto-index={{.*}}' only allowed with '-x ir'

; Ensure we get expected error for missing index file
; RUN: %clang -O2 -o %t4.o -x ir %t1.o -c -fthinlto-index=bad.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR1
; CHECK-ERROR1: Error loading index file 'bad.thinlto.bc'

; Ensure we ignore empty index file, and run non-ThinLTO compilation which
; would not import f2
; RUN: touch %t4.thinlto.bc
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t4.o -x ir %t1.o -c -fthinlto-index=%t4.thinlto.bc
; RUN: llvm-nm %t4.o | FileCheck --check-prefix=CHECK-OBJ-IGNORE-EMPTY %s
; CHECK-OBJ-IGNORE-EMPTY: T f1
; CHECK-OBJ-IGNORE-EMPTY: U f2

; Ensure we don't fail with index and non-ThinLTO object file, and output must
; be empty file.
; RUN: opt -o %t5.o %s
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t4.o -x ir %t5.o -c -fthinlto-index=%t.thinlto.bc
; RUN: llvm-nm %t4.o 2>&1 | count 0

; Ensure f2 was imported. Check for all 3 flavors of -save-temps[=cwd|obj].
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc -save-temps=obj
; RUN: llvm-dis %t1.s.3.import.bc -o - | FileCheck --check-prefix=CHECK-IMPORT %s
; RUN: mkdir -p %T/dir1
; RUN: cd %T/dir1
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc -save-temps=cwd
; RUN: cd ../..
; RUN: llvm-dis %T/dir1/*1.s.3.import.bc -o - | FileCheck --check-prefix=CHECK-IMPORT %s
; RUN: mkdir -p %T/dir2
; RUN: cd %T/dir2
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc -save-temps
; RUN: cd ../..
; RUN: llvm-dis %T/dir2/*1.s.3.import.bc -o - | FileCheck --check-prefix=CHECK-IMPORT %s
; CHECK-IMPORT: define available_externally void @f2()
; RUN: llvm-nm %t3.o | FileCheck --check-prefix=CHECK-OBJ %s
; CHECK-OBJ: T f1
; CHECK-OBJ-NOT: U f2

; Ensure we get expected error for input files without summaries
; RUN: opt -o %t2.o %s
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -o %t3.o -x ir %t1.o -c -fthinlto-index=%t.thinlto.bc 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR2
; CHECK-ERROR2: Error loading imported file '{{.*}}': Could not find module summary

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f2()
declare i8* @f3()

define void @f1() {
  call void @f2()
  ; Make sure that the backend can handle undefined references.
  ; Do an indirect call so that the undefined ref shows up in the combined index.
  call void bitcast (i8*()* @f3 to void()*)()
  ret void
}
