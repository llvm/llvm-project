// REQUIRES: x86-registered-target
/// When the input is a -fsplit-lto-unit bitcode file, link the regular LTO file like -mlink-bitcode-file.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm-bc -flto=thin -flto-unit -fsplit-lto-unit %s -o %t.bc
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-obj %t.bc -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %t.bc -o - | FileCheck %s --check-prefix=CHECK-IR

// CHECK:      V _ZTI1A
// CHECK-NEXT: V _ZTI1B
// CHECK-NEXT: V _ZTS1A
// CHECK-NEXT: V _ZTS1B
// CHECK-NEXT: V _ZTV1A
// CHECK-NEXT: V _ZTV1B

// CHECK-IR-DAG: _ZTS1B = linkonce_odr constant
// CHECK-IR-DAG: _ZTS1A = linkonce_odr constant
// CHECK-IR-DAG: _ZTV1B = linkonce_odr unnamed_addr constant
// CHECK-IR-DAG: _ZTI1A = linkonce_odr constant
// CHECK-IR-DAG: _ZTI1B = linkonce_odr constant
// CHECK-IR-DAG: _ZTV1A = linkonce_odr unnamed_addr constant

struct A {
  virtual int c(int i) = 0;
};

struct B : A {
  virtual int c(int i) { return i; }
};

int use() {
  return (new B)->c(0);
}
