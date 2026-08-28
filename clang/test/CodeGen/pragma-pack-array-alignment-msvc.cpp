// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -mlong-double-80 -emit-llvm -o - %s | FileCheck %s \
// RUN: --check-prefix=CHECK-FP80

// Test that #pragma pack does not reduce natural type alignment for vector
// and x86_fp80 types when used as array elements (matching MSVC behavior).

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

// Test x86_fp80 (long double with -mlong-double-80) arrays under pragma pack.
struct Klass { long double a; };

#pragma pack(push, 8)
template<typename T, unsigned N>
struct fp80_array {
  T _Elems[N];

  void fill(const T& val) {
    for (unsigned i = 0; i < N; i++)
      _Elems[i] = val;
  }
};
#pragma pack(pop)

// CHECK-FP80-LABEL: define {{.*}} @"?test_fp80_array
void test_fp80_array() {
  // CHECK-FP80: %matrix = alloca %struct.fp80_array, align 16
  fp80_array<Klass, 16> matrix;
  matrix.fill({});
}

// Struct containing x86_fp80 under pragma pack.
#pragma pack(push, 8)
struct Fp80Struct {
  long double val;
};

struct ArrayOfFp80Struct {
  Fp80Struct elems[4];
};
#pragma pack(pop)

// CHECK-FP80-LABEL: define {{.*}} @"?test_fp80_struct
void test_fp80_struct() {
  // CHECK-FP80: %s = alloca %struct.ArrayOfFp80Struct, align 16
  ArrayOfFp80Struct s;
  s.elems[0].val = 1.0L;
}
