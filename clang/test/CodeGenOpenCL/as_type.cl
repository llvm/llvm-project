// RUN: %clang_cc1 %s -emit-llvm -triple spir-unknown-unknown -o - | FileCheck %s

typedef __attribute__(( ext_vector_type(3) )) char char3;
typedef __attribute__(( ext_vector_type(4) )) char char4;
typedef __attribute__(( ext_vector_type(16) )) char char16;
typedef __attribute__(( ext_vector_type(3) )) int int3;

//CHECK: define spir_func <3 x i8> @f1(<4 x i8> %[[x:.*]])
//CHECK: %[[astype:.*]] = shufflevector <4 x i8> %[[x]], <4 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
//CHECK: ret <3 x i8> %[[astype]]
char3 f1(char4 x) {
  return  __builtin_astype(x, char3);
}

//CHECK: define spir_func <4 x i8> @f2(<3 x i8> %[[x:.*]])
//CHECK: %[[astype:.*]] = shufflevector <3 x i8> %[[x]], <3 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
//CHECK: ret <4 x i8> %[[astype]]
char4 f2(char3 x) {
  return __builtin_astype(x, char4);
}

//CHECK: define spir_func <3 x i8> @f3(i32 %[[x:.*]])
//CHECK: %[[cast:.*]] = bitcast i32 %[[x]] to <4 x i8>
//CHECK: %[[astype:.*]] = shufflevector <4 x i8> %[[cast]], <4 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
//CHECK: ret <3 x i8> %[[astype]]
char3 f3(int x) {
  return __builtin_astype(x, char3);
}

//CHECK: define spir_func <4 x i8> @f4(i32 %[[x:.*]])
//CHECK: %[[astype:.*]] = bitcast i32 %[[x]] to <4 x i8>
//CHECK-NOT: shufflevector
//CHECK: ret <4 x i8> %[[astype]]
char4 f4(int x) {
  return __builtin_astype(x, char4);
}

//CHECK: define spir_func i32 @f5(<3 x i8> %[[x:.*]])
//CHECK: %[[shuffle:.*]] = shufflevector <3 x i8> %[[x]], <3 x i8> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
//CHECK: %[[astype:.*]] = bitcast <4 x i8> %[[shuffle]] to i32
//CHECK: ret i32 %[[astype]]
int f5(char3 x) {
  return __builtin_astype(x, int);
}

//CHECK: define spir_func i32 @f6(<4 x i8> %[[x:.*]])
//CHECK: %[[astype:.*]] = bitcast <4 x i8> %[[x]] to i32
//CHECK-NOT: shufflevector
//CHECK: ret i32 %[[astype]]
int f6(char4 x) {
  return __builtin_astype(x, int);
}

//CHECK: define spir_func <3 x i8> @f7(<3 x i8> returned %[[x:.*]])
//CHECK-NOT: bitcast
//CHECK-NOT: shufflevector
//CHECK: ret <3 x i8> %[[x]]
char3 f7(char3 x) {
  return __builtin_astype(x, char3);
}

//CHECK: define spir_func <3 x i32> @f8(<16 x i8> %[[x:.*]])
//CHECK: %[[cast:.*]] = bitcast <16 x i8> %[[x]] to <4 x i32>
//CHECK: %[[astype:.*]] = shufflevector <4 x i32> %[[cast]], <4 x i32> undef, <3 x i32> <i32 0, i32 1, i32 2>
//CHECK: ret <3 x i32> %[[astype]]
int3 f8(char16 x) {
  return __builtin_astype(x, int3);
}
