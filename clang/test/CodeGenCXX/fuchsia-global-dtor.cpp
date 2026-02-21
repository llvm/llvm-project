/// Global destructors targetting Fuchsia should not use [__cxa_]atexit. Instead
/// they should be invoked through llvm.global_dtors.

// RUN: %clang_cc1 %s -triple aarch64-unknown-fuchsia -emit-llvm -o - | FileCheck %s

// CHECK-NOT: atexit

// CHECK:      @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }]
// CHECK-SAME:   [{ i32, ptr, ptr } { i32 {{.*}}, ptr [[MODULE_DTOR:@.*]], ptr {{.*}} }]

// CHECK:      define internal void [[MODULE_DTOR]]() {{.*}}{
// CHECK-NEXT: entry:
// CHECK-NEXT:   %0 = call ptr @_ZN1AD1Ev(ptr @DestroyFirst)
// CHECK-NEXT:   %1 = call ptr @_ZN1AD1Ev(ptr @DestroySecond)
// CHECK-NEXT:   %2 = call ptr @_ZN1AD1Ev(ptr @DestroyThird)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

struct A {
  ~A() {}
};

A DestroyThird;
A DestroySecond;
A DestroyFirst;
