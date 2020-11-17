// RUN: %clang_cc1 -triple x86_64 %s -fno-use-cxa-atexit -emit-llvm -o - | FileCheck %s

// PR7097
// RUN: %clang_cc1 -triple x86_64 %s -fno-use-cxa-atexit -mconstructor-aliases -emit-llvm -o - | FileCheck %s

// CHECK:      call void @_ZN1AC1Ev([[A:%.*]]* {{[^,]*}} @a)
// CHECK-NEXT: call i32 @atexit(void ()* @__dtor_a)
// CHECK:      define internal void @__dtor_a() [[NUW:#[0-9]+]]
// CHECK:      call void @_ZN1AD1Ev([[A]]* @a)

// CHECK:      call void @_ZN1AC1Ev([[A]]* {{[^,]*}} @b)
// CHECK-NEXT: call i32 @atexit(void ()* @__dtor_b)
// CHECK:      define internal void @__dtor_b() [[NUW]]
// CHECK:      call void @_ZN1AD1Ev([[A]]* @b)

class A {
public:
  A();
  ~A();
};

A a, b;

// PR9593
// CHECK-LABEL:      define void @_Z4funcv()
// CHECK:      call i32 @__cxa_guard_acquire(i64* @_ZGVZ4funcvE2a1)
// CHECK:      call void @_ZN1AC1Ev([[A]]* {{[^,]*}} @_ZZ4funcvE2a1)
// CHECK-NEXT: call i32 @atexit(void ()* @__dtor__ZZ4funcvE2a1)
// CHECK-NEXT: call void @__cxa_guard_release(i64* @_ZGVZ4funcvE2a1)

// CHECK:      call i32 @__cxa_guard_acquire(i64* @_ZGVZ4funcvE2a2)
// CHECK:      call void @_ZN1AC1Ev([[A]]* {{[^,]*}} @_ZZ4funcvE2a2)
// CHECK-NEXT: call i32 @atexit(void ()* @__dtor__ZZ4funcvE2a2)
// CHECK-NEXT: call void @__cxa_guard_release(i64* @_ZGVZ4funcvE2a2)

// CHECK:      define internal void @__dtor__ZZ4funcvE2a1() [[NUW]]
// CHECK:      call void @_ZN1AD1Ev([[A]]* @_ZZ4funcvE2a1)

// CHECK:      define internal void @__dtor__ZZ4funcvE2a2() [[NUW]]
// CHECK:      call void @_ZN1AD1Ev([[A]]* @_ZZ4funcvE2a2)

void func() {
  static A a1, a2;
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
