// RUN: %clang_cc1 -emit-llvm -fms-extensions %s -triple=x86_64-pc-windows-msvc -o - | FileCheck %s

// A `::delete` on a class whose deleting destructor is NOT defined in this TU
// (only declared, e.g. defined in another TU) must still emit a strong
// __global_delete forwarding body HERE. MSVC emits the forwarder at every
// `::delete` site (validated against cl.exe). Without this, a TU that only
// performs `::delete` (with the vector deleting destructor emitted in a
// different TU) would emit no forwarder, leaving the wrapper bound to the
// trapping __empty_global_delete fallback and crashing at runtime.

struct W {
  // Declared, but ~W() is defined in another TU, so no vector deleting
  // destructor (and hence no __global_delete wrapper reference) is emitted in
  // this TU. The forwarder must come from the ::delete site below.
  virtual ~W();
  void operator delete(void *);
  int x;
};

void sink(W *p) { ::delete p; }

// The shared trapping fallback is emitted (kept alive via llvm.used).
// CHECK: define linkonce_odr void @"?__empty_global_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @llvm.trap()
// CHECK-NEXT: unreachable

// The scalar wrapper gets a real forwarding body that calls the global
// ::operator delete (??3@), even though no deleting destructor in this TU
// references the wrapper.
// CHECK: define linkonce_odr void @"?__global_delete@@YAXPEAX_K@Z"(ptr noundef %0, i64 noundef %1)
// CHECK-NEXT: call void @"??3@YAXPEAX_K@Z"(ptr %0, i64 %1)

// No vector deleting destructor for W is emitted in this TU.
// CHECK-NOT: define {{.*}}@"??_EW@@
