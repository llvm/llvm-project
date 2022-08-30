// RUN: %clang_cc1 -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

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
// CHECK: call void {{.*}}AD2Ev
// CHECK: ret void

// After invoking base dtor and dtor for virtual base, poison vtable ptr.
// CHECK-LABEL: define {{.*}}BD1Ev
// CHECK: call void {{.*}}BD2Ev
// CHECK: call void {{.*}}AD2Ev
// CHECK: call void @__sanitizer_dtor_callback({{.*}}, !dbg ![[DI1:[0-9]+]]
// CHECK: ret void

// Since no virtual bases, poison vtable ptr here.
// CHECK-LABEL: define {{.*}}AD2Ev
// CHECK: call void @__sanitizer_dtor_callback({{.*}}, !dbg ![[DI2:[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback({{.*}}, !dbg ![[DI2]]
// CHECK: ret void

// Poison members
// CHECK-LABEL: define {{.*}}BD2Ev
// CHECK: call void @__sanitizer_dtor_callback({{.*}}, !dbg ![[DI4:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: !DIFile{{.*}}cpp

// CHECK-DAG: ![[DI1]] = {{.*}}line: [[@LINE-28]]
// CHECK-DAG: ![[DI2]] = {{.*}}line: [[@LINE-37]]
// CHECK-DAG: ![[DI4]] = {{.*}}line: [[@LINE-30]]
