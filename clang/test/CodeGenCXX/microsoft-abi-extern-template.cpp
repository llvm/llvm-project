// RUN: %clang_cc1 -fno-rtti-data -O1 -disable-llvm-optzns %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s

// Even though Foo<int> has an extern template declaration, we have to emit our
// own copy the vftable when emitting the available externally constructor.

// CHECK: @"\01??_7?$Foo@H@@6B@" = linkonce_odr unnamed_addr constant [1 x i8*] [
// CHECK-SAME:   i8* bitcast (i8* (%struct.Foo*, i32)* @"\01??_G?$Foo@H@@UEAAPEAXI@Z" to i8*)
// CHECK-SAME: ], comdat

// CHECK-LABEL: define %struct.Foo* @"\01?f@@YAPEAU?$Foo@H@@XZ"()
// CHECK: call %struct.Foo* @"\01??0?$Foo@H@@QEAA@XZ"(%struct.Foo* %{{.*}})

// CHECK: define available_externally %struct.Foo* @"\01??0?$Foo@H@@QEAA@XZ"(%struct.Foo* returned %this)
// CHECK:   store {{.*}} @"\01??_7?$Foo@H@@6B@"

// CHECK: define linkonce_odr i8* @"\01??_G?$Foo@H@@UEAAPEAXI@Z"(%struct.Foo* %this, i32 %should_call_delete)

struct Base {
  virtual ~Base();
};
template <typename T> struct Foo : Base {
  Foo() {}
};
extern template class Foo<int>;
Foo<int> *f() { return new Foo<int>(); }
