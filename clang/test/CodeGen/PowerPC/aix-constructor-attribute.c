// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s |\
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck %s

// CHECK: @llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo3, ptr null }, { i32, ptr, ptr } { i32 180, ptr @foo2, ptr null }, { i32, ptr, ptr } { i32 180, ptr @foo, ptr null }]

int foo(void) __attribute__((constructor(180)));
int foo2(void) __attribute__((constructor(180)));
int foo3(void) __attribute__((constructor(65535)));

int foo3(void) {
  return 3;
}

int foo2(void) {
  return 2;
}

int foo(void) {
  return 1;
}
