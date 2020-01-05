// RUN: %clang_cc1 -triple aarch64 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -fpatchable-function-entry=1 -o - | FileCheck --check-prefixes=CHECK,OPT %s

// CHECK: define void @f0() #0
__attribute__((patchable_function_entry(0))) void f0() {}

// CHECK: define void @f00() #0
__attribute__((patchable_function_entry(0, 0))) void f00() {}

// CHECK: define void @f2() #1
__attribute__((patchable_function_entry(2))) void f2() {}

// CHECK: define void @f20() #1
__attribute__((patchable_function_entry(2, 0))) void f20() {}

// CHECK: define void @f20decl() #1
__attribute__((patchable_function_entry(2, 0))) void f20decl();
void f20decl() {}

// OPT: define void @f() #2
void f() {}

/// M in patchable_function_entry(N,M) is currently ignored.
// CHECK: attributes #0 = { {{.*}} "patchable-function-entry"="0"
// CHECK: attributes #1 = { {{.*}} "patchable-function-entry"="2"
// OPT:   attributes #2 = { {{.*}} "patchable-function-entry"="1"
