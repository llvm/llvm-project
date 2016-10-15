// RUN: %clang_cc1 -emit-llvm -triple i686-windows-itanium -fdeclspec %s -o - | FileCheck %s

struct __declspec(dllexport) s {
  void f() {}
};

// CHECK: define {{.*}} dllexport {{.*}} @_ZN1saSERKS_
// CHECK: define {{.*}} dllexport {{.*}} @_ZN1s1fEv

