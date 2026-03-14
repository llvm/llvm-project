// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

void __attribute__((intel_ocl_bicc)) f1(void);

void f2(void) {
  f1();
// CHECK: call intel_ocl_bicc void @f1()
}

// CHECK: declare intel_ocl_bicc void @f1()
