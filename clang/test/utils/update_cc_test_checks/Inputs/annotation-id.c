// RUN: %clang_cc1 -ftrivial-auto-var-init=zero -triple=x86_64-unknown-linux-gnu -O1 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

void foo_ptr_to_scalar() {
  unsigned long long* a[100];
}
