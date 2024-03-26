; Test that ASan runs with the new pass manager
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address %s | FileCheck %s
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address -O1 %s | FileCheck %s

; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address -fprofile-instrument=llvm %s | FileCheck %s -check-prefixes=NOASAN
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address -fprofile-instrument=llvm -O1 %s | FileCheck %s -check-prefixes=NOASAN

; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address -fprofile-instrument=clang %s | FileCheck %s
; RUN: %clang_cc1 -triple x86_64-unknown-unknown -S -emit-llvm -o - -fsanitize=address -fprofile-instrument=clang -O1 %s | FileCheck %s

; CHECK-DAG: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(ptr %a) sanitize_address {
entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}

; CHECK: __asan_init

; CHECK-DAG: define internal void @asan.module_ctor() #[[#]] {
; CHECK:       {{.*}} call void @__asan_init()
; CHECK:       {{.*}} call void @__asan_version_mismatch_check_v8()
; CHECK:       ret void
; CHECK:     }

; CHECK-DAG: __asan_version_mismatch_check_v8

; NOASAN-NOT: __asan