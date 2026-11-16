// RUN: %clang -target aarch64-unknown-uefi -S -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=AARCH64_UEFI %s
// AARCH64_UEFI: target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
