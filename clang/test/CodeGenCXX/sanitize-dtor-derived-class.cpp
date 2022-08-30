// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

// Base dtor poisons members
// Complete dtor poisons vtable ptr after destroying members and
// virtual bases

class Base {
 public:
  int x;
  Base() {
    x = 5;
  }
  virtual ~Base() {
    x += 1;
  }
};

class Derived : public Base {
 public:
  int y;
  Derived() {
    y = 10;
  }
  ~Derived() {
    y += 1;
  }
};

Derived d;

// Invoke base destructor. No vtable pointer to poison.
// CHECK-LABEL: define {{.*}}DerivedD1Ev
// CHECK: call void {{.*}}DerivedD2Ev
// CHECK: ret void

// CHECK-LABEL: define {{.*}}DerivedD0Ev
// CHECK: call void {{.*}}DerivedD1Ev
// CHECK: ret void

// Invokes base destructor, and poison vtable pointer.
// CHECK-LABEL: define {{.*}}BaseD1Ev
// CHECK: call void {{.*}}BaseD2Ev
// CHECK: ret void

// CHECK-LABEL: define {{.*}}BaseD0Ev
// CHECK: call void {{.*}}BaseD1Ev
// CHECK: ret void

// Poison members and vtable ptr.
// CHECK-LABEL: define {{.*}}BaseD2Ev
// CHECK: call void @__sanitizer_dtor_callback{{.*}}, !dbg ![[DI1:[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback{{.*}}i64 8{{.*}}, !dbg ![[DI1]]
// CHECK: ret void

// Poison members and destroy non-virtual base.
// CHECK-LABEL: define {{.*}}DerivedD2Ev
// CHECK: call void @__sanitizer_dtor_callback{{.*}}, !dbg ![[DI3:[0-9]+]]
// CHECK: call void {{.*}}BaseD2Ev
// CHECK: call void @__sanitizer_dtor_callback{{.*}}i64 8{{.*}}, !dbg ![[DI3]]
// CHECK: ret void

// CHECK-LABEL: !DIFile{{.*}}cpp

// CHECK: ![[DI1]] = {{.*}}line: [[@LINE-49]]
// CHECK: ![[DI3]] = {{.*}}line: [[@LINE-39]]
