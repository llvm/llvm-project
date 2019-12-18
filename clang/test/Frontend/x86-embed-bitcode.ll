; REQUIRES: x86-registered-target
; check .ll input
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN:    | FileCheck %s
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=marker -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-MARKER
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ELF
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-bitcode=marker -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ELF-MARKER
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm \
; RUN:    -fembed-bitcode=bitcode -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ELF-ONLY-BITCODE

; check .bc input
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm-bc \
; RUN:    -x ir %s -o %t.bc
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %t.bc -o - \
; RUN:    | FileCheck %s
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=bitcode -x ir %t.bc -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ONLY-BITCODE
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=marker -x ir %t.bc -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-MARKER

; run through -fembed-bitcode twice and make sure it doesn't crash
; RUN: %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm-bc \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN: | %clang_cc1 -triple x86_64-apple-macosx10.10 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir - -o /dev/null

; check the magic number of bitcode at the beginning of the string
; CHECK: @llvm.embedded.module = private constant
; CHECK: c"\DE\C0\17\0B
; CHECK: section "__LLVM,__bitcode"
; CHECK: @llvm.cmdline = private constant
; CHECK: section "__LLVM,__cmdline"

; CHECK-ELF: @llvm.embedded.module
; CHECK-ELF: section ".llvmbc"
; CHECK-ELF: @llvm.cmdline
; CHECK-ELF: section ".llvmcmd"

; CHECK-ELF-MARKER: @llvm.embedded.module
; CHECK-ELF-MARKER: constant [0 x i8] zeroinitializer
; CHECK-ELF-MARKER: @llvm.cmdline
; CHECK-ELF-MARKER: section ".llvmcmd"

; CHECK-ELF-ONLY-BITCODE: @llvm.embedded.module
; CHECK-ELF-ONLY-BITCODE: section ".llvmbc"
; CHECK-ELF-ONLY-BITCODE-NOT: @llvm.cmdline
; CHECK-ELF-ONLY-BITCODE-NOT: section ".llvmcmd"

; CHECK-ONLY-BITCODE: @llvm.embedded.module = private constant
; CHECK-ONLY-BITCODE: c"\DE\C0\17\0B
; CHECK-ONLY-BITCODE: section "__LLVM,__bitcode"
; CHECK-ONLY-BITCODE-NOT: @llvm.cmdline = private constant
; CHECK-ONLY-BITCODE-NOT: section "__LLVM,__cmdline"

; CHECK-MARKER: @llvm.embedded.module
; CHECK-MARKER: constant [0 x i8] zeroinitializer
; CHECK-MARKER: section "__LLVM,__bitcode"
; CHECK-MARKER: @llvm.cmdline
; CHECK-MARKER: section "__LLVM,__cmdline"

define i32 @f0() {
  ret i32 0
}
