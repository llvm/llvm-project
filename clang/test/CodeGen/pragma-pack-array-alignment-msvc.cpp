// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -emit-llvm -o - %s | FileCheck %s

// Test that #pragma pack does not reduce natural type alignment for vector types
// when used as array elements (matching MSVC behavior).

typedef float __m128 __attribute__((__vector_size__(16)));

// Simple array-like struct with vector type under pragma pack.
#pragma pack(push, 8)
template<typename T, unsigned N>
struct array {
  T _Elems[N];
};
#pragma pack(pop)

// CHECK-LABEL: define {{.*}} @"?test_vector_array
void test_vector_array() {
  // CHECK: %matrix = alloca %struct.array, align 16
  array<__m128, 16> matrix;
  matrix._Elems[0] = (__m128){};
}

// Struct containing vector under pragma pack.
#pragma pack(push, 8)
struct VectorStruct {
  __m128 vec;
};

struct ArrayOfVectorStruct {
  VectorStruct elems[4];
};
#pragma pack(pop)

// CHECK-LABEL: define {{.*}} @"?test_vector_struct
void test_vector_struct() {
  // CHECK: %s = alloca %struct.ArrayOfVectorStruct, align 16
  ArrayOfVectorStruct s;
  s.elems[0].vec = (__m128){};
}
