; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -print-stats \
; RUN:    -emit-llvm -x ir %s -o - 2>&1 | FileCheck %s

; CHECK: *** Source Manager Stats
; CHECK: *** File Manager Stats
; CHECK: *** Virtual File System Stats
