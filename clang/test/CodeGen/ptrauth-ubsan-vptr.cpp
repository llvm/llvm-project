// RUN: %clang_cc1 -triple arm64e-apple-ios15 -fsanitize=vptr -O0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64e-apple-ios15 -fsanitize=vptr -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsanitize=vptr -O0 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsanitize=vptr -O2 -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

struct S {
  S() {}
  ~S() {}
  virtual int v() { return 0; }
  int a;
};

struct T : public S {
  virtual int v();
};

// CHECK-LABEL: foo1
int foo1(void* Buffer) {
  T *p = reinterpret_cast<T*>(Buffer);
  return p->v();
}
// CHECK-NOT: call {{.*}} @llvm.ptrauth.auth{{.*}}
// CHECK-NOT: call {{.*}} @llvm.ptrauth.strip{{.*}}

// CHECK-LABEL: foo2
int foo2(S* s) {
  T *p = dynamic_cast<T*>(s);
  return p->v();
}

// CHECK-NOT: call {{.*}} @llvm.ptrauth.auth{{.*}}
// CHECK-NOT: call {{.*}} @llvm.ptrauth.strip{{.*}}
