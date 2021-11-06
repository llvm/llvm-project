// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -emit-llvm -o - %s | FileCheck %s

struct Foo {
  char a;
  int b : 1;
};

int BitfieldOffset = sizeof(struct Foo);
// CHECK: @BitfieldOffset ={{.*}} global i32 2

int PointerSize = sizeof(void *);
// CHECK: @PointerSize ={{.*}} global i32 4

int PointerAlign = __alignof(void *);
// CHECK: @PointerAlign ={{.*}} global i32 4

int LongSize = sizeof(long);
// CHECK: @LongSize ={{.*}} global i32 4

int LongAlign = __alignof(long);
// CHECK: @LongAlign ={{.*}} global i32 4

// Not expected to change, but it's a difference between AAPCS and DarwinPCS
// that we need to be preserved for compatibility with ARMv7k.
long double LongDoubleVar = 0.0;
// CHECK: @LongDoubleVar ={{.*}} global double

typedef float __attribute__((ext_vector_type(16))) v16f32;
v16f32 func(v16f32 in) { return in; }
// CHECK: define{{.*}} void @func(<16 x float>* noalias sret(<16 x float>) align 16 {{%.*}}, <16 x float> noundef {{%.*}})
