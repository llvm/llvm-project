; RUN: %clang_cc1 -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=address %s | FileCheck %s

; CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor
; CHECK: declare void @__asan_loadN

define i32 @test_load(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
