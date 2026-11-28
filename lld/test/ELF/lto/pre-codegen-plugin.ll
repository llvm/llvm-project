; REQUIRES: x86, plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}

; RUN: opt %s -o %t.o
; RUN: ld.lld -shared --lto-emit-asm -%loadnewpmbye %t.o -o %t.inactive 2>&1 | count 0
; RUN: FileCheck %s --check-prefix=INACTIVE < %t.inactive.lto.s
; RUN: ld.lld -shared --lto-emit-asm -%loadnewpmbye --pass-plugin-arg=Bye,-last-words %t.o -o %t.active
; RUN: FileCheck %s --check-prefix=ACTIVE < %t.active.lto.s
; RUN: not ld.lld -shared -%loadnewpmbye --pass-plugin-arg=Bye,-last-words %t.o -o %t.err 2>&1 | FileCheck %s --check-prefix=ERR
; RUN: ld.lld -shared --lto-emit-asm --lto-partitions=2 -%loadnewpmbye --pass-plugin-arg=Bye,-last-words %t.o -o %t.parallel
; RUN: FileCheck %s --check-prefix=PARALLEL0 < %t.parallel.lto.s
; RUN: FileCheck %s --check-prefix=PARALLEL1 < %t.parallel.lto.1.s

;; Errors from loading plugins or checking their arguments are reported even without LTO
; RUN: llc %s -filetype=obj -o %t.native.o
; RUN: not ld.lld -shared --load-pass-plugin=%t.dir/nonexistent.so %t.native.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOADERR
; RUN: not ld.lld -shared -%loadnewpmbye --pass-plugin-arg=Nope,-x %t.native.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARGERR

; INACTIVE-NOT: Bye
; INACTIVE: f:
; ACTIVE: CodeGen Bye
; ERR: error: last words unsupported for binary output
; PARALLEL0: CodeGen Bye
; PARALLEL1: CodeGen Bye
; LOADERR: error: Could not load library {{.*}}nonexistent.so
; ARGERR: error: no pass plugin named 'Nope' is loaded

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @f(i32 %x) {
  ret i32 %x
}
