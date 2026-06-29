// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

// Verify that regular delete (not ::delete) does NOT trigger __global_delete
// forwarding body emission, but the VDD still uses __global_delete wrapper.
// This matches MSVC behavior where only ::delete triggers forwarding bodies.

struct Base {
  void* operator new(__SIZE_TYPE__);
  void operator delete(void*);
  void operator delete[](void*);
  virtual ~Base();
};
struct Derived : Base {
  virtual ~Derived();
};
Base::~Base() {}
Derived::~Derived() {}

// new[] forces VDD emission; regular delete[], not ::delete[].
void test() {
  Base *p = new Derived[2];
  delete[] p;
}

// The VDD dispatches between class and global delete using __global_delete.
// CHECK-LABEL: define weak dso_local noundef ptr @"??_EDerived@@UEAAPEAXI@Z"
// CHECK: dtor.call_glob_delete_after_array_destroy:
// CHECK: call void @"?__global_delete@@YAXPEAX_K@Z"(ptr noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK: dtor.call_glob_delete:
// CHECK-NEXT: call void @"?__global_delete@@YAXPEAX_K@Z"(ptr noundef %{{.*}}, i64 noundef 8)
// CHECK: dtor.call_class_delete:
// CHECK-NEXT: call void @"??3Base@@SAXPEAX@Z"(ptr noundef %{{.*}})

// __empty_global_delete should be emitted with a trap.
// CHECK: define linkonce_odr void @"?__empty_global_delete@@YAXPEAX_K@Z"(ptr %0, i64 %1)
// CHECK-NEXT: call void @llvm.trap()
// CHECK-NEXT: unreachable

// __global_delete should NOT have a forwarding body (no ::delete in this TU,
// no dllexport class).
// CHECK-NOT: define {{.*}}void @"?__global_delete@@YAXPEAX_K@Z"

// Verify the /ALTERNATENAME linker directive.
// CHECK: !{!"/alternatename:?__global_delete@@YAXPEAX_K@Z=?__empty_global_delete@@YAXPEAX_K@Z"}
