// RUN: %clang_cc1 -emit-llvm %s -o - -fms-extensions -triple=x86_64-pc-win32 -fexperimental-overflow-behavior-types | FileCheck %s

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __trap __attribute__((overflow_behavior(trap)))

typedef int __ob_wrap int_wrap;

// CHECK: define dso_local void @"?test_wrap_int@@YAXU?$ObtWrap_@H@__clang@@@Z"
void test_wrap_int(int_wrap x) {}

// CHECK: define dso_local void @"?test_trap_int@@YAXU?$ObtTrap_@H@__clang@@@Z"
void test_trap_int(int __ob_trap y) {}
