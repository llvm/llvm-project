// RUN: %clang_cc1 -fenable-matrix %s -emit-llvm -triple x86_64-unknown-linux -disable-llvm-passes -o - -std=c++11 | FileCheck %s

using i8x3 = _BitInt(8) __attribute__((ext_vector_type(3)));
using i8x3x3 = _BitInt(8) __attribute__((matrix_type(3, 3)));
using i32x3 = _BitInt(32) __attribute__((ext_vector_type(3)));
using i32x3x3 = _BitInt(32) __attribute__((matrix_type(3, 3)));
using i512x3 = _BitInt(512) __attribute__((ext_vector_type(3)));
using i512x3x3 = _BitInt(512) __attribute__((matrix_type(3, 3)));

// CHECK-LABEL: define dso_local i32 @_Z2v1Dv3_DB8_(i32 %a.coerce)
i8x3 v1(i8x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %retval = alloca <3 x i8>, align 4
  // CHECK-NEXT:   %a = alloca <3 x i8>, align 4
  // CHECK-NEXT:   %a.addr = alloca <3 x i8>, align 4
  // CHECK-NEXT:   store i32 %a.coerce, ptr %a, align 4
  // CHECK-NEXT:   %loadVec4 = load <4 x i8>, ptr %a, align 4
  // CHECK-NEXT:   %a1 = shufflevector <4 x i8> %loadVec4, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %extractVec = shufflevector <3 x i8> %a1, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  // CHECK-NEXT:   store <4 x i8> %extractVec, ptr %a.addr, align 4
  // CHECK-NEXT:   %loadVec42 = load <4 x i8>, ptr %a.addr, align 4
  // CHECK-NEXT:   %extractVec3 = shufflevector <4 x i8> %loadVec42, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %loadVec44 = load <4 x i8>, ptr %a.addr, align 4
  // CHECK-NEXT:   %extractVec5 = shufflevector <4 x i8> %loadVec44, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %add = add <3 x i8> %extractVec3, %extractVec5
  // CHECK-NEXT:   store <3 x i8> %add, ptr %retval, align 4
  // CHECK-NEXT:   %0 = load i32, ptr %retval, align 4
  // CHECK-NEXT:   ret i32 %0
  return a + a;
}

// CHECK-LABEL: define dso_local noundef <3 x i32> @_Z2v2Dv3_DB32_(<3 x i32> noundef %a)
i32x3 v2(i32x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %a.addr = alloca <3 x i32>, align 16
  // CHECK-NEXT:   %extractVec = shufflevector <3 x i32> %a, <3 x i32> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  // CHECK-NEXT:   store <4 x i32> %extractVec, ptr %a.addr, align 16
  // CHECK-NEXT:   %loadVec4 = load <4 x i32>, ptr %a.addr, align 16
  // CHECK-NEXT:   %extractVec1 = shufflevector <4 x i32> %loadVec4, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %loadVec42 = load <4 x i32>, ptr %a.addr, align 16
  // CHECK-NEXT:   %extractVec3 = shufflevector <4 x i32> %loadVec42, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %add = add <3 x i32> %extractVec1, %extractVec3
  // CHECK-NEXT:   ret <3 x i32> %add
  return a + a;
}

// CHECK-LABEL: define dso_local noundef <3 x i512> @_Z2v3Dv3_DB512_(ptr noundef byval(<3 x i512>) align 256 %0)
i512x3 v3(i512x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %a.addr = alloca <3 x i512>, align 256
  // CHECK-NEXT:   %loadVec4 = load <4 x i512>, ptr %0, align 256
  // CHECK-NEXT:   %a = shufflevector <4 x i512> %loadVec4, <4 x i512> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %extractVec = shufflevector <3 x i512> %a, <3 x i512> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  // CHECK-NEXT:   store <4 x i512> %extractVec, ptr %a.addr, align 256
  // CHECK-NEXT:   %loadVec41 = load <4 x i512>, ptr %a.addr, align 256
  // CHECK-NEXT:   %extractVec2 = shufflevector <4 x i512> %loadVec41, <4 x i512> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %loadVec43 = load <4 x i512>, ptr %a.addr, align 256
  // CHECK-NEXT:   %extractVec4 = shufflevector <4 x i512> %loadVec43, <4 x i512> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK-NEXT:   %add = add <3 x i512> %extractVec2, %extractVec4
  // CHECK-NEXT:   ret <3 x i512> %add
  return a + a;
}

// CHECK-LABEL: define dso_local noundef <9 x i8> @_Z2m1u11matrix_typeILm3ELm3EDB8_E(<9 x i8> noundef %a)
i8x3x3 m1(i8x3x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %a.addr = alloca [9 x i8], align 1
  // CHECK-NEXT:   store <9 x i8> %a, ptr %a.addr, align 1
  // CHECK-NEXT:   %0 = load <9 x i8>, ptr %a.addr, align 1
  // CHECK-NEXT:   %1 = load <9 x i8>, ptr %a.addr, align 1
  // CHECK-NEXT:   %2 = add <9 x i8> %0, %1
  // CHECK-NEXT:   ret <9 x i8> %2
  return a + a;
}

// CHECK-LABEL: define dso_local noundef <9 x i32> @_Z2m2u11matrix_typeILm3ELm3EDB32_E(<9 x i32> noundef %a)
i32x3x3 m2(i32x3x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %a.addr = alloca [9 x i32], align 4
  // CHECK-NEXT:   store <9 x i32> %a, ptr %a.addr, align 4
  // CHECK-NEXT:   %0 = load <9 x i32>, ptr %a.addr, align 4
  // CHECK-NEXT:   %1 = load <9 x i32>, ptr %a.addr, align 4
  // CHECK-NEXT:   %2 = add <9 x i32> %0, %1
  // CHECK-NEXT:   ret <9 x i32> %2
  return a + a;
}

// CHECK-LABEL: define dso_local noundef <9 x i512> @_Z2m3u11matrix_typeILm3ELm3EDB512_E(<9 x i512> noundef %a)
i512x3x3 m3(i512x3x3 a) {
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %a.addr = alloca [9 x i512], align 8
  // CHECK-NEXT:   store <9 x i512> %a, ptr %a.addr, align 8
  // CHECK-NEXT:   %0 = load <9 x i512>, ptr %a.addr, align 8
  // CHECK-NEXT:   %1 = load <9 x i512>, ptr %a.addr, align 8
  // CHECK-NEXT:   %2 = add <9 x i512> %0, %1
  // CHECK-NEXT:   ret <9 x i512> %2
  return a + a;
}
