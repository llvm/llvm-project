// RUN: %clang_cc1 -triple wasm32-unknown-unknown -target-feature +multivalue \
// RUN:   %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple wasm64-unknown-unknown -target-feature +multivalue \
// RUN:   %s -emit-llvm -o - | FileCheck %s

// Verify that the `wasm_multivalue` calling convention produces the function
// signatures described in the WebAssembly tool conventions PR.

// CHECK-LABEL: define wasm_multivalue void @f1()
__attribute__((wasm_multivalue))
void f1(void) {}

// CHECK-LABEL: define wasm_multivalue i32 @f2(float {{.*}}, double {{.*}})
__attribute__((wasm_multivalue))
int f2(float a, double b) { return (int)(a + b); }

// CHECK-LABEL: define wasm_multivalue i128 @f3(fp128
__attribute__((wasm_multivalue))
__int128 f3(long double x) { return (__int128)x; }

struct Foo4 { };
union Bar4 { };

// CHECK-LABEL: define wasm_multivalue void @f4()
__attribute__((wasm_multivalue))
union Bar4 f4(struct Foo4 x) { union Bar4 r; return r; }

struct Foo5 { int a; };
union Bar5 { int a; };

// CHECK-LABEL: define wasm_multivalue i32 @f5(i32
__attribute__((wasm_multivalue))
union Bar5 f5(struct Foo5 x) { union Bar5 r; r.a = x.a; return r; }

struct Foo6 { int a; int b; };

// CHECK-LABEL: define wasm_multivalue %struct.Foo6 @f6(i32 {{.*}}, i32 {{.*}})
__attribute__((wasm_multivalue))
struct Foo6 f6(struct Foo6 x) { return x; }

// CHECK-LABEL: define wasm_multivalue i128 @f7()
__attribute__((wasm_multivalue))
__int128 f7(void) { return 1; }

struct Foo8 { int a; int b; int c; };
// CHECK-LABEL: define wasm_multivalue %struct.Foo8 @f8(ptr
__attribute__((wasm_multivalue))
struct Foo8 f8(struct Foo8 x) { return x; }

struct Foo9 {
  struct Foo6 inner;
};
// CHECK-LABEL: define wasm_multivalue void @f9(ptr {{.*}} sret
__attribute__((wasm_multivalue))
struct Foo9 f9(void) { struct Foo9 r = {{0, 0}}; return r; }

// bitfields force pointers
struct Foo10 {
  int a : 4;
  int b : 4;
};
// CHECK-LABEL: define wasm_multivalue void @f10(ptr
__attribute__((wasm_multivalue))
struct Foo10 f10(void) { struct Foo10 r = {0, 0}; return r; }

// The default calling convention isn't changed from `+multivalue`
// CHECK-LABEL: define void @f11(ptr{{.*}}sret(%struct.Foo6){{.*}}, ptr {{.*}}byval(%struct.Foo6)
struct Foo6 f11(struct Foo6 x) { return x; }

// Test cross-calling-convention indierct calls
typedef __attribute__((wasm_multivalue)) struct Foo6 (*mv_ptr)(struct Foo6);

// CHECK-LABEL: define void @f12(
// CHECK: call wasm_multivalue {{(noundef )?}}%struct.Foo6 %0(i32{{.*}}, i32
struct Foo6 f12(mv_ptr fn, struct Foo6 x) {
  return fn(x);
}

struct Foo13 {
  int empty_array[0];
};

// CHECK-LABEL: define wasm_multivalue void @f13()
__attribute__((wasm_multivalue))
struct Foo13 f13(struct Foo13 x) {
  return x;
}

struct Foo14 {
  int one_element_array[1];
};

// CHECK-LABEL: define wasm_multivalue i32 @f14(i32
__attribute__((wasm_multivalue))
struct Foo14 f14(struct Foo14 x) {
  return x;
}

struct Foo15 {
  int two_element_array[2];
};

// CHECK-LABEL: define wasm_multivalue void @f15(ptr {{.*}}, ptr {{.*}})
__attribute__((wasm_multivalue))
struct Foo15 f15(struct Foo15 x) {
  return x;
}

struct Foo16 {
  int three_element_array[3];
};

// CHECK-LABEL: define wasm_multivalue void @f16(ptr {{.*}}, ptr {{.*}})
__attribute__((wasm_multivalue))
struct Foo16 f16(struct Foo16 x) {
  return x;
}

// CHECK-LABEL: define wasm_multivalue { double, double } @complex_types(double {{.*}}, double {{.*}})
__attribute__((wasm_multivalue))
_Complex double complex_types(_Complex double x) {
  return x;
}
