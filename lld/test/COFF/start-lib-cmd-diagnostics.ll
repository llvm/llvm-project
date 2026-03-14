; REQUIRES: x86
;
; We need an input file to lld, so create one.
; RUN: llc -filetype=obj %s -o %t.obj

; RUN: not lld-link %t.obj -end-lib 2>&1 \
; RUN:     | FileCheck --check-prefix=STRAY_END %s
; STRAY_END: stray -end-lib

; RUN: not lld-link -start-lib -start-lib %t.obj 2>&1 \
; RUN:     | FileCheck --check-prefix=NESTED_START %s
; NESTED_START: nested -start-lib

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

define void @main() {
  ret void
}
