; REQUIRES: x86-registered-target

; Test that the ThinLTO backend reads its files through -vfsoverlay if provided.

; RUN: rm -rf %t && mkdir -p %t/orig %t/remote
; RUN: opt -module-summary -o %t/orig/1.o %s
; RUN: opt -module-summary -o %t/orig/2.o %S/Inputs/thinlto_backend.ll
; RUN: llvm-lto -thinlto -o %t/orig/all %t/orig/1.o %t/orig/2.o
; RUN: mv %t/orig/1.o %t/orig/2.o %t/orig/all.thinlto.bc %t/remote

; No overlay provided, the index file is not found.
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -c -x ir %t/remote/1.o \
; RUN:   -fthinlto-index=%t/orig/all.thinlto.bc -o %t/none.o 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-NONE
; CHECK-NONE: Error loading index file

; Overlay of the index only, the import is still missing.
; RUN: echo "{'version': 0, 'use-external-names': false, 'roots': [{'type': 'file', 'name': '%/t/orig/all.thinlto.bc', 'external-contents': '%/t/remote/all.thinlto.bc'}]}" > %t/one-file.yaml
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -c -x ir %t/remote/1.o \
; RUN:   -fthinlto-index=%t/orig/all.thinlto.bc -vfsoverlay %t/one-file.yaml \
; RUN:   -o %t/one-file.o 2>&1 | FileCheck %s --check-prefix=CHECK-ONE
; CHECK-ONE: Error loading imported file

; Remap the directory so f2 is imported and f1 stays for either external names.
; RUN: echo "{'version': 0, 'use-external-names': false, 'roots': [{'type': 'directory-remap', 'name': '%/t/orig', 'external-contents': '%/t/remote'}]}" > %t/vfs.yaml
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -c -x ir %t/orig/1.o \
; RUN:   -fthinlto-index=%t/orig/all.thinlto.bc -vfsoverlay %t/vfs.yaml \
; RUN:   -o %t/remapped.o
; RUN: llvm-nm %t/remapped.o | FileCheck %s --check-prefix=CHECK-OBJ
; RUN: echo "{'version': 0, 'use-external-names': true, 'roots': [{'type': 'directory-remap', 'name': '%/t/orig', 'external-contents': '%/t/remote'}]}" > %t/vfs-external.yaml
; RUN: %clang -target x86_64-unknown-linux-gnu -O2 -c -x ir %t/orig/1.o \
; RUN:   -fthinlto-index=%t/orig/all.thinlto.bc -vfsoverlay %t/vfs-external.yaml \
; RUN:   -o %t/remapped-external.o
; RUN: llvm-nm %t/remapped-external.o | FileCheck %s --check-prefix=CHECK-OBJ
; CHECK-OBJ: T f1
; CHECK-OBJ-NOT: U f2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f2()

define void @f1() {
  call void @f2()
  ret void
}
