// RUN: %clang_cc1 -fzos-extensions -emit-llvm -triple s390x-ibm-zos -x c++ -o - %s | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define void @_Z2f1v()
void f1() {}

// CHECK-LABEL: define void @_Z2f2Pi(ptr addrspace(1) noundef %p32)
void f2(int * __ptr32 p32) {}

// CHECK-LABEL: define noundef ptr addrspace(1) @_Z2f3Pi(ptr addrspace(1) noundef %p32)
int * __ptr32 f3(int * __ptr32 p32) {
  return p32;
}

// CHECK-LABEL: define noundef ptr @_Z2f4PPi(ptr noundef %p32)
int * __ptr32 *f4(int * __ptr32 *p32) {
  return p32;
}
