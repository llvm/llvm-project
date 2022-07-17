; REQUIRES: x86-macho-wonkiness

;; Creating read-only directories with `chmod 400` isn't supported on Windows
; UNSUPPORTED: system-windows

;; -object_path_lto specifies a directory that cannot be created
; RUN: rm -rf %t && mkdir %t && mkdir %t/dir
; RUN: chmod 400 %t/dir
; RUN: llvm-as %s -o %t/full.o
; RUN: not %lld  %t/full.o -o /dev/null -object_path_lto %t/dir/dir2 2>&1 | FileCheck %s --check-prefix=READONLY -DDIR=%t/dir/dir2

; READONLY: error: cannot create LTO object path [[DIR]]: {{.*}}

;; Multiple objects need to be created, but -object_path_lto doesn't point to a directory
; RUN: touch %t/out.o
; RUN: opt -module-summary %s -o %t/thin.o
; RUN: not %lld %t/full.o %t/thin.o -o /dev/null -object_path_lto %t/out.o 2>&1 | FileCheck %s --check-prefix=MULTIPLE

; MULTIPLE: error: -object_path_lto must specify a directory when using ThinLTO


target triple = "x86_64-apple-macosx10.15.0"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @main() {
  ret void
}
