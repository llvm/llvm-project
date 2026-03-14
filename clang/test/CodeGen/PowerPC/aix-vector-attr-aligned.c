// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec -target-cpu pwr7 -emit-llvm -o - %s | \
// RUN:   FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec -target-cpu pwr7 -emit-llvm -o - %s | \
// RUN:   FileCheck %s

typedef vector int __attribute__((aligned(8))) UnderAlignedVI;

vector int g32 __attribute__((aligned(32)));
vector int g8 __attribute__((aligned(8)));
UnderAlignedVI TypedefedGlobal;

int escape(vector int*);

int local32(void) {
  vector int l32 __attribute__((aligned(32)));
  return escape(&l32);
}

int local8(void) {
  vector int l8 __attribute__((aligned(8)));
  return escape(&l8);
}

// CHECK: @g32 = global <4 x i32> zeroinitializer, align 32
// CHECK: @g8 = global <4 x i32> zeroinitializer, align 16
// CHECK: @TypedefedGlobal = global <4 x i32> zeroinitializer, align 8

// CHECK-LABEL: @local32
// CHECK:         %l32 = alloca <4 x i32>, align 32
//
// CHECK-LABEL: @local8
// CHECK:         %l8 = alloca <4 x i32>, align 16
