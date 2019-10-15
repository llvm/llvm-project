; Test that ASan runs with the new pass manager
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=address %s | FileCheck %s --check-prefixes=CHECK,LTO,THINLTO
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=address -flto %s | FileCheck %s --check-prefixes=CHECK,LTO
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fexperimental-new-pass-manager -fsanitize=address -flto=thin %s | FileCheck %s --check-prefixes=CHECK,THINLTO
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=address %s | FileCheck %s --check-prefixes=CHECK,LTO,THINLTO
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=address -flto %s | FileCheck %s --check-prefixes=CHECK,LTO
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -O1 -fexperimental-new-pass-manager -fsanitize=address -flto=thin %s | FileCheck %s --check-prefixes=CHECK,THINLTO

; DAG-CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

; CHECK: __asan_init

; DAG-CHECK: define internal void @asan.module_ctor() {
; CHECK:       {{.*}} call void @__asan_init()
; CHECK:       {{.*}} call void @__asan_version_mismatch_check_v8()
; CHECK:       ret void
; CHECK:     }

; DAG-CHECK: __asan_version_mismatch_check_v8

; This is not used in ThinLTO
; DAG-LTO: __asan_report_load4
