// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=CHECK32 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck -check-prefix=CHECK64 %s

struct A {
  long x, y;
};

struct B {
  long x, y, z, w;
};

extern "C" {

int f1(A, A, A, A);
B f2(void);
_Complex float f3(void);
A &f4();

}

void test() {
  A a;

  // CHECK32: call i32 @f1(ptr nest noundef @f1
  // CHECK64: call i32 @f1(ptr nest noundef @f1
  __builtin_call_with_static_chain(f1(a, a, a, a), f1);

  // CHECK32: call void @f2(ptr sret(%struct.B) align 4 %{{[0-9a-z]+}}, ptr nest noundef @f2)
  // CHECK64: call void @f2(ptr sret(%struct.B) align 8 %{{[0-9a-z]+}}, ptr nest noundef @f2)
  __builtin_call_with_static_chain(f2(), f2);

  // CHECK32: call i64 @f3(ptr nest noundef @f3)
  // CHECK64: call <2 x float> @f3(ptr nest noundef @f3)
  __builtin_call_with_static_chain(f3(), f3);

  // CHECK32: call nonnull align 4 dereferenceable(8) ptr @f4(ptr nest noundef @f4)
  // CHECK64: call nonnull align 8 dereferenceable(16) ptr @f4(ptr nest noundef @f4)
  __builtin_call_with_static_chain(f4(), f4);
}
