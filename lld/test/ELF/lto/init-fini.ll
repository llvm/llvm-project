; REQUIRES: x86
; RUN: llvm-as %s -o %t.o

;;
;; Verify that symbols given by -init and -fini are preserved and
;; DT_INIT/DT_FINI are created.
;;

; RUN: ld.lld -o %t.exe -pie %t.o
; RUN: llvm-nm %t.exe | FileCheck -check-prefix=TEST1 --allow-empty %s
; RUN: llvm-readelf -d %t.exe | FileCheck -check-prefix=TEST2 %s

; TEST1-NOT: foo
; TEST1-NOT: bar

; TEST2-NOT: INIT
; TEST2-NOT: FINI

; RUN: ld.lld -o %t.exe -pie -init=foo -fini=bar %t.o
; RUN: llvm-nm %t.exe | FileCheck -check-prefix=TEST3 %s
; RUN: llvm-readelf -d %t.exe | FileCheck -check-prefix=TEST4 %s

; TEST3: bar
; TEST3: foo

; TEST4: INIT
; TEST4: FINI

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}
