// RUN: %clang_cc1 -fno-rtti-data -O1 -disable-llvm-passes %s -emit-llvm -o - -triple x86_64-windows-msvc | FileCheck %s

// Even though Foo<int> has an extern template declaration, we have to emit our
// own copy the vftable when emitting the available externally constructor.

// CHECK: @"??_7?$Foo@H@@6B@" = linkonce_odr unnamed_addr constant { [1 x ptr] } { [1 x ptr] [
// CHECK-SAME:   ptr @"??_G?$Foo@H@@UEAAPEAXI@Z"
// CHECK-SAME: ] }, comdat

// CHECK-LABEL: define dso_local noundef ptr @"?f@@YAPEAU?$Foo@H@@XZ"()
// CHECK: call noundef ptr @"??0?$Foo@H@@QEAA@XZ"(ptr {{[^,]*}} %{{.*}})

// CHECK: define available_externally dso_local noundef ptr @"??0?$Foo@H@@QEAA@XZ"(ptr {{[^,]*}} returned align 8 dereferenceable(8) %this)
// CHECK:   store {{.*}} @"??_7?$Foo@H@@6B@"

// CHECK: define linkonce_odr dso_local noundef ptr @"??_G?$Foo@H@@UEAAPEAXI@Z"(ptr {{[^,]*}} %this, i32 noundef %should_call_delete)

struct Base {
  virtual ~Base();
};
template <typename T> struct Foo : Base {
  Foo() {}
};
extern template class Foo<int>;
Foo<int> *f() { return new Foo<int>(); }
