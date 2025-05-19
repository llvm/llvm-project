// RUN: %clang -target x86_64-unknown-uefi -S -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=X86_64_UEFI %s
// X86_64_UEFI: target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
