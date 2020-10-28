; REQUIRES: arm-registered-target
; REQUIRES: aarch64-registered-target
; check .ll input
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN:    | FileCheck %s
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=marker -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-MARKER
; RUN: %clang_cc1 -triple aarch64-unknown-linux-gnueabi -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ELF

; check .bc input
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm-bc \
; RUN:    -x ir %s -o %t.bc
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %t.bc -o - \
; RUN:    | FileCheck %s
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=bitcode -x ir %t.bc -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-ONLY-BITCODE
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=marker -x ir %t.bc -o - \
; RUN:    | FileCheck %s -check-prefix=CHECK-MARKER

; run through -fembed-bitcode twice and make sure it doesn't crash
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm-bc \
; RUN:    -fembed-bitcode=all -x ir %s -o - \
; RUN: | %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir - -o /dev/null

; check the magic number of bitcode at the beginning of the string
; CHECK: @llvm.embedded.module = private constant
; CHECK: c"\DE\C0\17\0B
; CHECK: section "__LLVM,__bitcode"
; CHECK: @llvm.cmdline = private constant
; CHECK: section "__LLVM,__cmdline"

; check warning options are not embedded
; RUN: %clang_cc1 -triple thumbv7-apple-ios8.0.0 -emit-llvm \
; RUN:    -fembed-bitcode=all -x ir %s -o - -Wall -Wundef-prefix=TEST \
; RUN:    | FileCheck %s -check-prefix=CHECK-WARNING

; CHECK-ELF: @llvm.embedded.module
; CHECK-ELF-SAME: section ".llvmbc", align 1
; CHECK-ELF: @llvm.cmdline
; CHECK-ELF-SAME: section ".llvmcmd", align 1

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

; CHECK-WARNING-NOT: Wall
; CHECK-WARNING-NOT: Wundef-prefix

define i32 @f0() {
  ret i32 0
}
