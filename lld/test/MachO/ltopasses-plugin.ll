; REQUIRES: x86, plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}

; RUN: opt %s -o %t.o
; RUN: %lld -dylib -%loadnewpmbye %t.o -o %t.inactive 2>&1 | count 0

; RUN: not %lld -dylib -%loadnewpmbye --pass-plugin-arg=Bye,-last-words %t.o -o %t.err 2>&1 | FileCheck %s --check-prefix=ERR

;; Errors from loading plugins or checking their arguments are reported even without LTO
; RUN: llc %s -filetype=obj -o %t.native.o
; RUN: not %lld -dylib --load-pass-plugin=%t.dir/nonexistent.dylib %t.native.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOADERR
; RUN: not %lld -dylib -%loadnewpmbye --pass-plugin-arg=Nope,-x %t.native.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARGERR

; ERR: error: last words unsupported for binary output
; LOADERR: error: Could not load library {{.*}}nonexistent.dylib
; ARGERR: error: no pass plugin named 'Nope' is loaded

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

define i32 @f(i32 %x) {
  ret i32 %x
}
