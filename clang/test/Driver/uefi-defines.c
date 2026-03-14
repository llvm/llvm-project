// RUN: %clang -target x86_64-unknown-uefi %s -emit-llvm -S -c -o - | FileCheck %s

// CHECK: __UEFI__defined
#ifdef __UEFI__
void __UEFI__defined() {}
#endif
