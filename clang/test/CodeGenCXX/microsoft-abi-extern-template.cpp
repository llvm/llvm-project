// RUN: %clang_cc1 -fno-rtti-data -O1 -disable-llvm-passes %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s

// Even though Foo<int> has an extern template declaration, we have to emit our
// own copy the vftable when emitting the available externally constructor.

// CHECK: @"??_7?$Foo@H@@6B@" = linkonce_odr unnamed_addr constant { [1 x i8*] } { [1 x i8*] [
// CHECK-SAME:   i8* bitcast (i8* (%struct.Foo*, i32)* @"??_G?$Foo@H@@UEAAPEAXI@Z" to i8*)
// CHECK-SAME: ] }, comdat

// CHECK-LABEL: define dso_local noundef %struct.Foo* @"?f@@YAPEAU?$Foo@H@@XZ"()
// CHECK: call noundef %struct.Foo* @"??0?$Foo@H@@QEAA@XZ"(%struct.Foo* {{[^,]*}} %{{.*}})

// CHECK: define available_externally dso_local noundef %struct.Foo* @"??0?$Foo@H@@QEAA@XZ"(%struct.Foo* {{[^,]*}} returned %this)
// CHECK:   store {{.*}} @"??_7?$Foo@H@@6B@"

// CHECK: define linkonce_odr dso_local noundef i8* @"??_G?$Foo@H@@UEAAPEAXI@Z"(%struct.Foo* {{[^,]*}} %this, i32 noundef %should_call_delete)

struct Base {
  virtual ~Base();
};
template <typename T> struct Foo : Base {
  Foo() {}
};
extern template class Foo<int>;
Foo<int> *f() { return new Foo<int>(); }
