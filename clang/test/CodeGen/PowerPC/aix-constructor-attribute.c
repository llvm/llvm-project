// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s |\
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck %s

// CHECK: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo3, ptr null }, { i32, ptr, ptr } { i32 180, ptr @foo2, ptr null }, { i32, ptr, ptr } { i32 180, ptr @foo, ptr null }]

void foo(void) __attribute__((constructor(180)));
void foo2(void) __attribute__((constructor(180)));
void foo3(void) __attribute__((constructor(65535)));

void foo3(void) {}
void foo2(void) {}
void foo(void) {}
