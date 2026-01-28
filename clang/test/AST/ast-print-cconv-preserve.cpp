// RUN: %clang_cc1 -ast-print -triple x86_64-unknown-unknown %s -o - | FileCheck %s

void (__attribute__((preserve_none)) *none)();

// CHECK: __attribute__((preserve_none)) void (*none)();

__attribute__((preserve_all)) void (*all)();

// CHECK: __attribute__((preserve_all)) void ((*all))();

__attribute__((preserve_most)) void (*most)();

// CHECK: __attribute__((preserve_most)) void ((*most))();

