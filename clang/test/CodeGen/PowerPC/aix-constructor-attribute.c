// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s |\
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck %s

// CHECK: @llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* bitcast (i32 ()* @foo3 to void ()*), i8* null }, { i32, void ()*, i8* } { i32 180, void ()* bitcast (i32 ()* @foo2 to void ()*), i8* null }, { i32, void ()*, i8* } { i32 180, void ()* bitcast (i32 ()* @foo to void ()*), i8* null }]

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
