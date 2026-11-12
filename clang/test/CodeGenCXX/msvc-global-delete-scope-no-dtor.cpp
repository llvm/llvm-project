// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - \
// RUN:   | FileCheck %s --implicit-check-not='define{{.*}}"?__global_array_delete@@YAXPEAX_K@Z"'

// A `::delete` that does not run a non-trivial destructor must NOT cause a
// __global_delete forwarding body to be emitted, even when the TU also
// contains a class whose vector deleting destructor routes its global-delete
// path through __global_delete.
//
// This matches MSVC (validated against cl.exe 19.44): MSVC only engages the
// __global_delete machinery when a deleting destructor is involved, i.e. for a
// `::delete` on a class type with a non-trivial destructor. A `::delete` on a
// primitive or on a trivially-destructible class is lowered as a plain direct
// operator delete, so MSVC leaves __global_delete as a weak external that falls
// back to the trapping __empty_global_delete. Clang models this with a weak
// alias to __empty_global_delete. (The destructor's virtualness and whether the
// class has its own operator delete are irrelevant to this trigger.) Emitting a
// forwarding body here would add a hard reference to the global operator delete
// and could reintroduce LNK2001 in environments without one.

struct Base {
  void *operator new[](__SIZE_TYPE__);
  void operator delete[](void *);
  virtual ~Base();
};
struct Derived : Base {
  virtual ~Derived();
};
Base::~Base() {}
Derived::~Derived() {}

// Forces emission of Derived's vector deleting destructor, which creates the
// __global_delete wrapper (as a declaration) for its global-delete path.
void makeVDD() {
  Base *p = new Derived[2];
  delete[] p;
}

// Explicit global-scope delete on a primitive: no destructor involved, so it
// must not trigger forwarding-body emission.
void scopePrimitive(int *q) {
  ::delete q;
}

// Explicit global-scope delete on a trivially-destructible class: still no
// (non-trivial) destructor to run, so it must not trigger it either.
struct Trivial {
  int x;
};
void scopeTrivial(Trivial *q) {
  ::delete q;
}

// The __global_array_delete wrapper defaults to a weak alias to the trapping
// fallback (no forwarding body of its own is emitted, asserted via the
// --implicit-check-not on the RUN line). The alias is emitted before the
// function definitions.
// CHECK: @"?__global_array_delete@@YAXPEAX_K@Z" = weak alias void (ptr, i64), ptr @"?__empty_global_delete@@YAXPEAX_K@Z"

// The vector deleting destructor still routes its global array-delete path
// through the __global_array_delete wrapper.
// CHECK: call void @"?__global_array_delete@@YAXPEAX_K@Z"

// __empty_global_delete is emitted as the trapping fallback (shared by the
// scalar and array wrappers of this signature).
// CHECK: define linkonce_odr void @"?__empty_global_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @llvm.trap()
// CHECK-NEXT: unreachable
