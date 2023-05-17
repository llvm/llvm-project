// RUN: %clang_cc1 -fsanitize=memory -O0 -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -fsanitize=memory -O1 -disable-llvm-passes -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

template <class T>
class Vector {
public:
  int size;
  ~Vector() {
    size += 1;
  }
};

struct Base {
  int b1;
  double b2;
  Base() {
    b1 = 5;
    b2 = 10.989;
  }
  virtual ~Base() {}
};

struct VirtualBase {
  int vb1;
  int vb2;
  VirtualBase() {
    vb1 = 10;
    vb2 = 11;
  }
  virtual ~VirtualBase() {}
};

struct Derived : public Base, public virtual VirtualBase {
  int d1;
  Vector<int> v;
  int d2;
  Derived() {
    d1 = 10;
  }
  ~Derived() {}
};

Derived d;

// Destruction order:
// Derived: int, Vector, Base, VirtualBase

// CHECK-LABEL: define {{.*}}ZN7DerivedD1Ev
// CHECK: call void {{.*}}ZN11VirtualBaseD2Ev
// CHECK: call void @__sanitizer_dtor_callback_vptr({{.*}}){{.*}}, !dbg ![[DI0:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: define {{.*}}ZN7DerivedD0Ev
// CHECK: ret void

// CHECK-LABEL: define {{.*}}ZN11VirtualBaseD1Ev
// CHECK: ret void

// CHECK-LABEL: define {{.*}}ZN11VirtualBaseD0Ev
// CHECK: ret void

// poison 2 ints
// CHECK-LABEL: define {{.*}}ZN11VirtualBaseD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 8){{.*}}, !dbg ![[DI1:[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback_vptr({{.*}}){{.*}}{{.*}}, !dbg ![[DI2:[0-9]+]]
// CHECK: ret void

// poison int and double
// CHECK-LABEL: define {{.*}}ZN4BaseD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 16){{.*}}, !dbg ![[DI3:[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback_vptr({{.*}}{{.*}}, !dbg ![[DI4:[0-9]+]]
// CHECK: ret void

// poison int, ignore vector, poison int
// CHECK-LABEL: define {{.*}}ZN7DerivedD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 4){{.*}}, !dbg ![[DI5:[0-9]+]]
// CHECK: call void {{.*}}ZN6VectorIiED1Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 4){{.*}}, !dbg ![[DI6:[0-9]+]]
// CHECK: call void {{.*}}ZN4BaseD2Ev
// CHECK: ret void

// poison int
// CHECK-LABEL: define {{.*}}ZN6VectorIiED2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 4){{.*}}, !dbg ![[DI7:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: !DIFile{{.*}}.cpp

// CHECK-DAG: ![[DI0]] = {{.*}}line: [[@LINE-49]]
// CHECK-DAG: ![[DI1]] = {{.*}}line: [[@LINE-66]]
// CHECK-DAG: ![[DI2]] = {{.*}}line: [[@LINE-61]]
// CHECK-DAG: ![[DI3]] = {{.*}}line: [[@LINE-78]]
// CHECK-DAG: ![[DI5]] = {{.*}}line: [[@LINE-57]]
// CHECK-DAG: ![[DI6]] = {{.*}}line: [[@LINE-60]]
// CHECK-DAG: ![[DI7]] = {{.*}}line: [[@LINE-88]]
