// Test AllocToken respects ignorelist for functions and files.
//
// RUN: %clang_cc1 -fsanitize=alloc-token -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-ALLOW
//
// RUN: echo "fun:excluded_by_all" > %t.func.ignorelist
// RUN: %clang_cc1 -fsanitize=alloc-token -fsanitize-ignorelist=%t.func.ignorelist -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-FUN
//
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t.file.ignorelist
// RUN: %clang_cc1 -fsanitize=alloc-token -fsanitize-ignorelist=%t.file.ignorelist -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SRC

extern void* malloc(unsigned long size);

// CHECK-LABEL: define{{.*}} @excluded_by_all(
void* excluded_by_all(unsigned long size) {
    // CHECK-ALLOW: call ptr @__alloc_token_malloc(
    // CHECK-FUN: call ptr @malloc(
    // CHECK-SRC: call ptr @malloc(
    return malloc(size);
}

// CHECK-LABEL: define{{.*}} @excluded_by_src(
void* excluded_by_src(unsigned long size) {
    // CHECK-ALLOW: call ptr @__alloc_token_malloc(
    // CHECK-FUN: call ptr @__alloc_token_malloc(
    // CHECK-SRC: call ptr @malloc(
    return malloc(size);
}
