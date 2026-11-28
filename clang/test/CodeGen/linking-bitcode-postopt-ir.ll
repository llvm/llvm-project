; Check that -mlink-builtin-bitcode-postopt links the builtin bitcode after the
; optimizations when the input is LLVM IR, as it does for source input.

; RUN: split-file %s %t
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm-bc \
; RUN:   -o %t/lib.bc -x ir %t/lib.ll
; RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm \
; RUN:   -mlink-builtin-bitcode %t/lib.bc -mlink-builtin-bitcode-postopt \
; RUN:   -o - -x ir %t/main.ll | FileCheck %s

; CHECK: define {{.*}}i32 @main()
; CHECK: call i32 @lib_func(i32 1)
; CHECK: define {{.*}}i32 @lib_func(i32

;--- main.ll
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %r = call i32 @lib_func(i32 1)
  ret i32 %r
}

declare i32 @lib_func(i32)

;--- lib.ll
target triple = "x86_64-unknown-linux-gnu"

define i32 @lib_func(i32 %x) noinline {
entry:
  %r = add i32 %x, 41
  ret i32 %r
}
