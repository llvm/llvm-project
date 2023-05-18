; RUN: %clang_cc1 -flto -triple x86_64-pc-linux-gnu -emit-llvm-bc -disable-llvm-passes -x ir < %s -o - | llvm-bcanalyzer -dump | FileCheck %s
; RUN: %clang_cc1 -flto=thin -triple x86_64-pc-linux-gnu -emit-llvm-bc -disable-llvm-passes -x ir < %s -o - | llvm-bcanalyzer -dump | FileCheck %s
; REQUIRES: x86-registered-target

; CHECK-NOT:GLOBALVAL_SUMMARY_BLOCK

; Make sure this doesn't crash, and we don't try to emit a module summary.
; (The command is roughly emulating what -save-temps would do.)
@0 = global i32 0
