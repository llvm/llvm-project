// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s |\
// RUN:   FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck %s

// CHECK: @llvm.global_ctors = appending global [4 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* bitcast (i32 ()* @_Z4foo3v to void ()*), i8* null }, { i32, void ()*, i8* } { i32 180, void ()* bitcast (i32 ()* @_Z4foo2v to void ()*), i8* null }, { i32, void ()*, i8* } { i32 180, void ()* bitcast (i32 ()* @_Z3foov to void ()*), i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I__, i8* null }]
// CHECK: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__D_a, i8* null }]

int foo() __attribute__((constructor(180)));
int foo2() __attribute__((constructor(180)));
int foo3() __attribute__((constructor(65535)));

struct Test {
public:
  Test() {}
  ~Test() {}
} t;

int foo3() {
  return 3;
}

int foo2() {
  return 2;
}

int foo() {
  return 1;
}
