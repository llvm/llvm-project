// RUN: %clang_cc1 -triple powerpc-ibm-aix -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -emit-llvm -o - %s | FileCheck %s

class a {
  virtual void d() {}
  virtual void e();
};
void a::e() {}

// CHECK-NOT: = comdat
