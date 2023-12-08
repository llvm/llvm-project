; Verify that in the cases of explict distinct LTO piplines,
; explicit unified LTO pipelines, and the default LTO pipeline,
; there is no crash and the anonoymous global is named
; as expected.

; RUN: %clang_cc1 -emit-llvm-bc -O1 -flto -fsanitize=address -o - -x ir < %s | llvm-dis -o - | FileCheck %s
; RUN: %clang_cc1 -emit-llvm-bc -O1 -flto -funified-lto -fsanitize=address -o - -x ir < %s | llvm-dis -o - | FileCheck %s
; CHECK: @anon.3ee0898e5200a57350fed5485ae5d237

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"none\00", align 1

define ptr @f() {
  %ptr = getelementptr inbounds [5 x i8], ptr @.str, i32 0, i32 0
  ret ptr %ptr
}
