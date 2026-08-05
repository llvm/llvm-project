// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

// Verify that a plain delete (no `::`) does NOT trigger __global_delete
// forwarding body emission, but the VDD still uses the __global_delete wrapper.
// This matches MSVC (validated against cl.exe): a plain delete that resolves to
// a global operator delete never emits a forwarding body; only an explicit
// `::delete` on a class type does.

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

// Each wrapper defaults to a weak alias to the trapping __empty_global_delete
// (matching MSVC's weak-external-with-default; this also works under Arm64EC,
// unlike an /alternatename directive). These aliases are emitted before the
// function definitions.
// CHECK-DAG: @"?__global_delete@@YAXPEAX_K@Z" = weak alias void (ptr, i64), ptr @"?__empty_global_delete@@YAXPEAX_K@Z"
// CHECK-DAG: @"?__global_array_delete@@YAXPEAX_K@Z" = weak alias void (ptr, i64), ptr @"?__empty_global_delete@@YAXPEAX_K@Z"

// The VDD dispatches between class and global delete: the array path uses the
// __global_array_delete wrapper, the scalar path uses __global_delete.
// CHECK-LABEL: define weak dso_local noundef ptr @"??_EDerived@@UEAAPEAXI@Z"
// CHECK: dtor.call_glob_delete_after_array_destroy:
// CHECK: call void @"?__global_array_delete@@YAXPEAX_K@Z"(ptr noundef %{{.*}}, i64 noundef %{{.*}})
// CHECK: dtor.call_glob_delete:
// CHECK-NEXT: call void @"?__global_delete@@YAXPEAX_K@Z"(ptr noundef %{{.*}}, i64 noundef 8)
// CHECK: dtor.call_class_delete:
// CHECK-NEXT: call void @"??3Base@@SAXPEAX@Z"(ptr noundef %{{.*}})

// __empty_global_delete should be emitted with a trap.
// CHECK: define linkonce_odr void @"?__empty_global_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @llvm.trap()
// CHECK-NEXT: unreachable

// Neither wrapper should have a forwarding body (no `::delete` on a class
// type in this TU, and no dllexport class).
// CHECK-NOT: define {{.*}}void @"?__global_delete@@YAXPEAX_K@Z"
// CHECK-NOT: define {{.*}}void @"?__global_array_delete@@YAXPEAX_K@Z"
