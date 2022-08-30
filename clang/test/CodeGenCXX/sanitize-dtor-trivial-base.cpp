// RUN: %clang_cc1 -no-opaque-pointers -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -no-opaque-pointers -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

// Base class has trivial dtor => complete dtor poisons base class memory directly.

class Base {
public:
  int x[4];
};

class Derived : public Base {
public:
  int y;
  ~Derived() {
  }
};

Derived d;

// Poison members, then poison the trivial base class.
// CHECK-LABEL: define {{.*}}DerivedD2Ev
// CHECK: %[[GEP:[0-9a-z]+]] = getelementptr i8, i8* {{.*}}, i64 16
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}%[[GEP]], i64 4{{.*}}, !dbg ![[DI1:[0-9]+]]
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}, i64 16{{.*}}, !dbg ![[DI2:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: !DIFile{{.*}}cpp

// CHECK-DAG: ![[DI1]] = {{.*}}line: [[@LINE-16]]
// CHECK-DAG: ![[DI2]] = {{.*}}line: [[@LINE-24]]
