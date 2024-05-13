// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s |\
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -x c++ -emit-llvm \
// RUN:     -fno-use-cxa-atexit < %s | \
// RUN:   FileCheck %s

// CHECK: @llvm.global_ctors = appending global [4 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_Z4foo3v, ptr null }, { i32, ptr, ptr } { i32 180, ptr @_Z4foo2v, ptr null }, { i32, ptr, ptr } { i32 180, ptr @_Z3foov, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I__, ptr null }]
// CHECK: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__D_a, ptr null }]

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
