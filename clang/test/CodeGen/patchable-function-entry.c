// RUN: %clang_cc1 -triple aarch64 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -fpatchable-function-entry=1 -o - | FileCheck --check-prefixes=CHECK,OPT %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -fms-hotpatch -o - | FileCheck --check-prefixes=HOTPATCH %s

// CHECK: define{{.*}} void @f0() #0
__attribute__((patchable_function_entry(0))) void f0() {}

// CHECK: define{{.*}} void @f00() #0
__attribute__((patchable_function_entry(0, 0))) void f00() {}

// CHECK: define{{.*}} void @f2() #1
__attribute__((patchable_function_entry(2))) void f2() {}

// CHECK: define{{.*}} void @f20() #1
__attribute__((patchable_function_entry(2, 0))) void f20() {}

// CHECK: define{{.*}} void @f20decl() #1
__attribute__((patchable_function_entry(2, 0))) void f20decl();
void f20decl() {}

// CHECK: define{{.*}} void @f44() #2
__attribute__((patchable_function_entry(4, 4))) void f44() {}

// CHECK: define{{.*}} void @f52() #3
__attribute__((patchable_function_entry(5, 2))) void f52() {}

// OPT: define{{.*}} void @f() #4
void f() {}

/// No need to emit "patchable-function-entry"="0"
// CHECK: attributes #0 = { {{.*}}
// CHECK-NOT: "patchable-function-entry"

// CHECK: attributes #1 = { {{.*}} "patchable-function-entry"="2"
// CHECK: attributes #2 = { {{.*}} "patchable-function-entry"="0" "patchable-function-prefix"="4"
// CHECK: attributes #3 = { {{.*}} "patchable-function-entry"="3" "patchable-function-prefix"="2"
// OPT:   attributes #4 = { {{.*}} "patchable-function-entry"="1"
// HOTPATCH: attributes #0 = { {{.*}} "patchable-function"="prologue-short-redirect"
// HOTPATCH: attributes #1 = { {{.*}} "patchable-function"="prologue-short-redirect"
// HOTPATCH: attributes #2 = { {{.*}} "patchable-function"="prologue-short-redirect"
// HOTPATCH: attributes #3 = { {{.*}} "patchable-function"="prologue-short-redirect"
