; REQUIRES: x86

; RUN: rm -rf %t; mkdir %t
; RUN: llvm-as %s -o %t/full.o
; RUN: %lld -dylib %t/full.o -o %t/full.dylib \
; RUN:   --opt-remarks-filename %t/full.yaml \
; RUN:   --opt-remarks-passes inline \
; RUN:   --opt-remarks-with-hotness \
; RUN:   --opt-remarks-hotness-threshold 300
; RUN: FileCheck %s --check-prefix=REMARK < %t/full.yaml
; RUN: %lld -dylib %t/full.o -o %t/threshold.dylib \
; RUN:   --opt-remarks-filename %t/threshold.yaml \
; RUN:   --opt-remarks-with-hotness \
; RUN:   --opt-remarks-hotness-threshold 301
; RUN: count 0 < %t/threshold.yaml
; RUN: %lld -dylib %t/full.o -o %t/filtered.dylib \
; RUN:   --opt-remarks-filename %t/filtered.yaml \
; RUN:   --opt-remarks-passes does-not-match
; RUN: count 0 < %t/filtered.yaml

; RUN: opt -module-summary %s -o %t/thin.o
; RUN: %lld -dylib %t/thin.o -o %t/thin.dylib \
; RUN:   --opt-remarks-filename=%t/thin.yaml \
; RUN:   --opt-remarks-passes=inline \
; RUN:   --opt-remarks-format=yaml
; RUN: cat %t/thin.yaml.thin.*.yaml | FileCheck %s --check-prefix=THIN

; RUN: not %lld -dylib %t/full.o -o /dev/null \
; RUN:   --opt-remarks-hotness-threshold invalid 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ERR
; RUN: not %lld -dylib %t/full.o -o /dev/null \
; RUN:   --opt-remarks-hotness-threshold=invalid 2>&1 | \
; RUN:   FileCheck %s --check-prefix=ERR

; REMARK:      --- !Passed
; REMARK:      Pass:            inline
; REMARK:      Name:            Inlined
; REMARK:      Function:        caller
; REMARK:      Hotness:         300
; REMARK:      Callee:          callee
; REMARK:      Caller:          caller

; THIN:        --- !Passed
; THIN:        Pass:            inline
; THIN:        Name:            Inlined
; THIN:        Function:        caller
; THIN:        Callee:          callee
; THIN:        Caller:          caller

; ERR: --opt-remarks-hotness-threshold: invalid argument 'invalid', only integer or 'auto' is supported

target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @callee(i32 %x) {
  %add = add i32 %x, 1
  ret i32 %add
}

define i32 @caller(i32 %x) !prof !0 {
  %result = call i32 @callee(i32 %x)
  ret i32 %result
}

!0 = !{!"function_entry_count", i64 300}
