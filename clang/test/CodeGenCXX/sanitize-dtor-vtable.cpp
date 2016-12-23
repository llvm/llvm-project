// RUN: %clang_cc1 -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

class A {
 public:
  int x;
  A() {}
  virtual ~A() {}
};
A a;

class B : virtual public A {
 public:
  int y;
  B() {}
  ~B() {}
};
B b;

// CHECK-LABEL: define {{.*}}AD1Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: call void {{.*}}AD2Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void

// After invoking base dtor and dtor for virtual base, poison vtable ptr.
// CHECK-LABEL: define {{.*}}BD1Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: call void {{.*}}BD2Ev
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: call void {{.*}}AD2Ev
// CHECK: call void @__sanitizer_dtor_callback{{.*}}i64 8
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void

// Since no virtual bases, poison vtable ptr here.
// CHECK-LABEL: define {{.*}}AD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: call void @__sanitizer_dtor_callback{{.*}}i64 8
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void

// Poison members
// CHECK-LABEL: define {{.*}}BD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK-NOT: call void @__sanitizer_dtor_callback
// CHECK: ret void
