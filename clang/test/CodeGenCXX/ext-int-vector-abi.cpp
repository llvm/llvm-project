// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN64
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-gnu-linux -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=LIN32

// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN64
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i386-windows-pc -O3 -disable-llvm-passes -I%S -emit-llvm -o - %s | FileCheck %s --check-prefixes=WIN32

// Make sure BitInt vector match builtin Int vector abi.

using int8_t3 = _BitInt(8)  __attribute__((ext_vector_type(3)));
int8_t3 ManglingTestRetParam(int8_t3 Param) {
// LIN64: define{{.*}} i32 @_Z20ManglingTestRetParamDv3_DB8_(i32 %
// LIN32: define{{.*}} <3 x i8> @_Z20ManglingTestRetParamDv3_DB8_(<3 x i8> %
// WIN64: define dso_local <3 x i8> @"?ManglingTestRetParam@@YAT?$__vector@U?$_BitInt@$07@__clang@@$02@__clang@@T12@@Z"(<3 x i8> %
// WIN32: define dso_local <3 x i8> @"?ManglingTestRetParam@@YAT?$__vector@U?$_BitInt@$07@__clang@@$02@__clang@@T12@@Z"(<3 x i8> inreg %
  return Param;
}
using int8_t3c = char  __attribute__((ext_vector_type(3)));
int8_t3c ManglingTestRetParam(int8_t3c Param) {
// LIN64: define{{.*}} i32 @_Z20ManglingTestRetParamDv3_c(i32 %
// LIN32: define{{.*}} <3 x i8> @_Z20ManglingTestRetParamDv3_c(<3 x i8> %
// WIN64: define dso_local <3 x i8> @"?ManglingTestRetParam@@YAT?$__vector@D$02@__clang@@T12@@Z"(<3 x i8> %
// WIN32: define dso_local <3 x i8> @"?ManglingTestRetParam@@YAT?$__vector@D$02@__clang@@T12@@Z"(<3 x i8> inreg %
  return Param;
}

typedef unsigned _BitInt(16) uint16_t4 __attribute__((ext_vector_type(4)));
uint16_t4 ManglingTestRetParam(uint16_t4 Param) {
// LIN64: define{{.*}} double @_Z20ManglingTestRetParamDv4_DU16_(double %
// LIN32: define{{.*}} <4 x i16> @_Z20ManglingTestRetParamDv4_DU16_(i64 %
// WIN64: define dso_local <4 x i16> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@T12@@Z"(<4 x i16> %
// WIN32: define dso_local <4 x i16> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0BA@@__clang@@$03@__clang@@T12@@Z"(<4 x i16> inreg %
  return Param;
}

typedef unsigned short uint16_t4s __attribute__((ext_vector_type(4)));
uint16_t4s ManglingTestRetParam(uint16_t4s Param) {
// LIN64: define{{.*}} double @_Z20ManglingTestRetParamDv4_t(double %
// LIN32: define{{.*}} <4 x i16> @_Z20ManglingTestRetParamDv4_t(i64 %
// WIN64: define dso_local <4 x i16> @"?ManglingTestRetParam@@YAT?$__vector@G$03@__clang@@T12@@Z"(<4 x i16> %
// WIN32: define dso_local <4 x i16> @"?ManglingTestRetParam@@YAT?$__vector@G$03@__clang@@T12@@Z"(<4 x i16> inreg %
  return Param;
}

typedef unsigned _BitInt(32) uint32_t4 __attribute__((ext_vector_type(4)));
uint32_t4 ManglingTestRetParam(uint32_t4 Param) {
// LIN64: define{{.*}} <4 x i32> @_Z20ManglingTestRetParamDv4_DU32_(<4 x i32> %
// LIN32: define{{.*}} <4 x i32> @_Z20ManglingTestRetParamDv4_DU32_(<4 x i32> %
// WIN64: define dso_local <4 x i32> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0CA@@__clang@@$03@__clang@@T12@@Z"(<4 x i32> %
// WIN32: define dso_local <4 x i32> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0CA@@__clang@@$03@__clang@@T12@@Z"(<4 x i32> inreg %
  return Param;
}

typedef unsigned int uint32_t4s __attribute__((ext_vector_type(4)));
uint32_t4s ManglingTestRetParam(uint32_t4s Param) {
// LIN64: define{{.*}} <4 x i32> @_Z20ManglingTestRetParamDv4_j(<4 x i32> %
// LIN32: define{{.*}} <4 x i32> @_Z20ManglingTestRetParamDv4_j(<4 x i32> %
// WIN64: define dso_local <4 x i32> @"?ManglingTestRetParam@@YAT?$__vector@I$03@__clang@@T12@@Z"(<4 x i32> %
// WIN32: define dso_local <4 x i32> @"?ManglingTestRetParam@@YAT?$__vector@I$03@__clang@@T12@@Z"(<4 x i32> inreg %
  return Param;
}

typedef unsigned _BitInt(64) uint64_t4 __attribute__((ext_vector_type(4)));
uint64_t4 ManglingTestRetParam(uint64_t4 Param) {
// LIN64: define{{.*}} <4 x i64> @_Z20ManglingTestRetParamDv4_DU64_(ptr byval(<4 x i64>) align 32 %
// LIN32: define{{.*}} <4 x i64> @_Z20ManglingTestRetParamDv4_DU64_(<4 x i64> %
// WIN64: define dso_local <4 x i64> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0EA@@__clang@@$03@__clang@@T12@@Z"(<4 x i64> %
// WIN32: define dso_local <4 x i64> @"?ManglingTestRetParam@@YAT?$__vector@U?$_UBitInt@$0EA@@__clang@@$03@__clang@@T12@@Z"(<4 x i64> inreg %
  return Param;
}

typedef unsigned long long uint64_t4s __attribute__((ext_vector_type(4)));
uint64_t4s ManglingTestRetParam(uint64_t4s Param) {
// LIN64: define{{.*}} <4 x i64> @_Z20ManglingTestRetParamDv4_y(ptr byval(<4 x i64>) align 32 %
// LIN32: define{{.*}} <4 x i64> @_Z20ManglingTestRetParamDv4_y(<4 x i64> %
// WIN64: define dso_local <4 x i64> @"?ManglingTestRetParam@@YAT?$__vector@_K$03@__clang@@T12@@Z"(<4 x i64> %
// WIN32: define dso_local <4 x i64> @"?ManglingTestRetParam@@YAT?$__vector@_K$03@__clang@@T12@@Z"(<4 x i64> inreg %
  return Param;
}

typedef _BitInt(32) vint32_t8 __attribute__((vector_size(32)));
vint32_t8 ManglingTestRetParam(vint32_t8 Param) {
// LIN64: define{{.*}} <8 x i32> @_Z20ManglingTestRetParamDv8_DB32_(ptr byval(<8 x i32>) align 32 %
// LIN32: define{{.*}} <8 x i32> @_Z20ManglingTestRetParamDv8_DB32_(<8 x i32> %
// WIN64: define dso_local <8 x i32> @"?ManglingTestRetParam@@YA?AT?$__vector@U?$_BitInt@$0CA@@__clang@@$07@__clang@@T12@@Z"(<8 x i32> %
// WIN32: define dso_local <8 x i32> @"?ManglingTestRetParam@@YA?AT?$__vector@U?$_BitInt@$0CA@@__clang@@$07@__clang@@T12@@Z"(<8 x i32> inreg %
  return Param;
}

typedef int vint32_t8i __attribute__((vector_size(32)));
vint32_t8i ManglingTestRetParam(vint32_t8i Param) {
// LIN64: define{{.*}} <8 x i32> @_Z20ManglingTestRetParamDv8_i(ptr byval(<8 x i32>) align 32 %
// LIN32: define{{.*}} <8 x i32> @_Z20ManglingTestRetParamDv8_i(<8 x i32> %
// WIN64: define dso_local <8 x i32> @"?ManglingTestRetParam@@YA?AT?$__vector@H$07@__clang@@T12@@Z"(<8 x i32> %
// WIN32: define dso_local <8 x i32> @"?ManglingTestRetParam@@YA?AT?$__vector@H$07@__clang@@T12@@Z"(<8 x i32> inreg %
  return Param;
}

typedef unsigned _BitInt(64) uvint64_t16 __attribute__((vector_size(16)));
uvint64_t16 ManglingTestRetParam(uvint64_t16 Param) {
// LIN64: define{{.*}} <2 x i64> @_Z20ManglingTestRetParamDv2_DU64_(<2 x i64> %
// LIN32: define{{.*}} <2 x i64> @_Z20ManglingTestRetParamDv2_DU64_(<2 x i64> %
// WIN64: define dso_local <2 x i64> @"?ManglingTestRetParam@@YA?AT?$__vector@U?$_UBitInt@$0EA@@__clang@@$01@__clang@@T12@@Z"(<2 x i64> %
// WIN32: define dso_local <2 x i64> @"?ManglingTestRetParam@@YA?AT?$__vector@U?$_UBitInt@$0EA@@__clang@@$01@__clang@@T12@@Z"(<2 x i64> inreg %
  return Param;
}
using uvint64_t16l = unsigned long long  __attribute__((vector_size(16)));
uvint64_t16l ManglingTestRetParam(uvint64_t16l Param) {
// LIN64: define{{.*}} <2 x i64> @_Z20ManglingTestRetParamDv2_y(<2 x i64> %
// LIN32: define{{.*}} <2 x i64> @_Z20ManglingTestRetParamDv2_y(<2 x i64> %
// WIN64: define dso_local <2 x i64> @"?ManglingTestRetParam@@YA?AT?$__vector@_K$01@__clang@@T12@@Z"(<2 x i64> %
// WIN32: define dso_local <2 x i64> @"?ManglingTestRetParam@@YA?AT?$__vector@_K$01@__clang@@T12@@Z"(<2 x i64> inreg %
  return Param;
}
