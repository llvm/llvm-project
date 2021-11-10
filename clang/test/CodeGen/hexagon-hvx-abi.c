// RUN: %clang_cc1 -triple hexagon -emit-llvm -target-cpu hexagonv66 -target-feature +hvxv66 -target-feature +hvx-length64b -o - %s | FileCheck %s --check-prefix CHECK-HVX64
// RUN: %clang_cc1 -triple hexagon -emit-llvm -target-cpu hexagonv66 -target-feature +hvxv66 -target-feature +hvx-length128b -o - %s | FileCheck %s --check-prefix CHECK-HVX128

typedef long HVX_Vector __attribute__((__vector_size__(__HVX_LENGTH__)))
  __attribute__((aligned(__HVX_LENGTH__)));
typedef long HVX_VectorPair __attribute__((__vector_size__(2*__HVX_LENGTH__)))
  __attribute__((aligned(__HVX_LENGTH__)));

// CHECK-HVX64: define {{.*}} <16 x i32> @foo(<16 x i32> %a, <32 x i32> %b)
// CHECK-HVX128: define {{.*}} <32 x i32> @foo(<32 x i32> %a, <64 x i32> %b) 
HVX_Vector foo(HVX_Vector a, HVX_VectorPair b) {
  return a;
}

// CHECK-HVX64: define {{.*}} <32 x i32> @bar(<16 x i32> %a, <32 x i32> %b)
// CHECK-HVX128: define {{.*}} <64 x i32> @bar(<32 x i32> %a, <64 x i32> %b)
HVX_VectorPair bar(HVX_Vector a, HVX_VectorPair b) {
  return b;
}

