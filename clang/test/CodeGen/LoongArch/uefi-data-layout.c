// RUN: %clang -target loongarch64-unknown-uefi -S -emit-llvm -o - %s | \
// RUN:     FileCheck --check-prefix=LA64_UEFI %s
// LA64_UEFI: target datalayout = "e-m:w-p:64:64-i64:64-i128:128-n32:64-S128"
