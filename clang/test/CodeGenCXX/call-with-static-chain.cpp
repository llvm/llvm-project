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

  // CHECK32: call i32 bitcast (i32 (i32, i32, i32, i32, i32, i32, i32, i32)* @f1 to i32 (i8*, i32, i32, i32, i32, i32, i32, i32, i32)*)(i8* nest bitcast (i32 (i32, i32, i32, i32, i32, i32, i32, i32)* @f1 to i8*)
  // CHECK64: call i32 bitcast (i32 (i64, i64, i64, i64, i64, i64, %struct.A*)* @f1 to i32 (i8*, i64, i64, i64, i64, i64, i64, %struct.A*)*)(i8* nest bitcast (i32 (i64, i64, i64, i64, i64, i64, %struct.A*)* @f1 to i8*)
  __builtin_call_with_static_chain(f1(a, a, a, a), f1);

  // CHECK32: call void bitcast (void (%struct.B*)* @f2 to void (%struct.B*, i8*)*)(%struct.B* sret(%struct.B) align 4 %{{[0-9a-z]+}}, i8* nest bitcast (void (%struct.B*)* @f2 to i8*))
  // CHECK64: call void bitcast (void (%struct.B*)* @f2 to void (%struct.B*, i8*)*)(%struct.B* sret(%struct.B) align 8 %{{[0-9a-z]+}}, i8* nest bitcast (void (%struct.B*)* @f2 to i8*))
  __builtin_call_with_static_chain(f2(), f2);

  // CHECK32: call i64 bitcast (i64 ()* @f3 to i64 (i8*)*)(i8* nest bitcast (i64 ()* @f3 to i8*))
  // CHECK64: call <2 x float> bitcast (<2 x float> ()* @f3 to <2 x float> (i8*)*)(i8* nest bitcast (<2 x float> ()* @f3 to i8*))
  __builtin_call_with_static_chain(f3(), f3);

  // CHECK32: call nonnull align 4 dereferenceable(8) %struct.A* bitcast (%struct.A* ()* @f4 to %struct.A* (i8*)*)(i8* nest bitcast (%struct.A* ()* @f4 to i8*))
  // CHECK64: call nonnull align 8 dereferenceable(16) %struct.A* bitcast (%struct.A* ()* @f4 to %struct.A* (i8*)*)(i8* nest bitcast (%struct.A* ()* @f4 to i8*))
  __builtin_call_with_static_chain(f4(), f4);
}
