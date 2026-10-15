// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -ffreestanding -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -ffreestanding -mlong-double-80 -emit-llvm -o - %s | FileCheck %s \
// RUN: --check-prefix=CHECK-FP80

// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fms-extensions \
// RUN: -ffreestanding -mlong-double-80 -fdump-record-layouts -fsyntax-only %s \
// RUN: | FileCheck %s --check-prefix=CHECK-LAYOUT

// Test that #pragma pack does not reduce natural type alignment for vector
// and x86_fp80 types when used as array elements (matching MSVC behavior).

#include <xmmintrin.h>

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
  matrix._Elems[0] = _mm_setzero_ps();
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
  s.elems[0].vec = _mm_setzero_ps();
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

// An alignment attribute on a typedef of x86_fp80 makes the alignment
// requirement come from the typedef rather than from the pragma-pack-resistant
// natural alignment of the underlying type.

// The typedef raises the alignment above the natural alignment.
struct TypedefOverAligned {
  typedef long double foo __attribute__((aligned(32)));
  foo bar;
};

// CHECK-LAYOUT-LABEL:   0 | struct TypedefOverAligned{{$}}
// CHECK-LAYOUT-NEXT:    0 |   foo bar
// CHECK-LAYOUT-NEXT:      | [sizeof=32, align=32,
// CHECK-LAYOUT-NEXT:      |  nvsize=32, nvalign=32]

// The typedef's alignment attribute is not reduced by #pragma pack.
#pragma pack(push, 8)
struct TypedefOverAlignedPacked {
  char c;
  typedef long double foo __attribute__((aligned(32)));
  foo bar;
};
#pragma pack(pop)

// CHECK-LAYOUT-LABEL:   0 | struct TypedefOverAlignedPacked{{$}}
// CHECK-LAYOUT-NEXT:    0 |   char c
// CHECK-LAYOUT-NEXT:   32 |   foo bar
// CHECK-LAYOUT-NEXT:      | [sizeof=64, align=32,
// CHECK-LAYOUT-NEXT:      |  nvsize=48, nvalign=32]

// The typedef lowers the alignment below the natural alignment, so the field
// no longer resists #pragma pack.
#pragma pack(push, 8)
typedef long double fp80_align4 __attribute__((aligned(4)));
struct TypedefUnderAlignedPacked {
  char c;
  fp80_align4 x;
};
#pragma pack(pop)

// CHECK-LAYOUT-LABEL:   0 | struct TypedefUnderAlignedPacked{{$}}
// CHECK-LAYOUT-NEXT:    0 |   char c
// CHECK-LAYOUT-NEXT:    8 |   fp80_align4 x
// CHECK-LAYOUT-NEXT:      | [sizeof=24, align=8,
// CHECK-LAYOUT-NEXT:      |  nvsize=24, nvalign=8]

// A plain x86_fp80 field resists #pragma pack.
#pragma pack(push, 8)
struct PlainFp80Packed {
  char c;
  long double x;
};
#pragma pack(pop)

// CHECK-LAYOUT-LABEL:   0 | struct PlainFp80Packed{{$}}
// CHECK-LAYOUT-NEXT:    0 |   char c
// CHECK-LAYOUT-NEXT:   16 |   long double x
// CHECK-LAYOUT-NEXT:      | [sizeof=32, align=16,
// CHECK-LAYOUT-NEXT:      |  nvsize=32, nvalign=16]

// A typedef without an alignment attribute keeps the pragma-pack resistance.
#pragma pack(push, 8)
typedef long double fp80_t;
struct TypedefNoAttrPacked {
  char c;
  fp80_t x;
};
#pragma pack(pop)

// CHECK-LAYOUT-LABEL:   0 | struct TypedefNoAttrPacked{{$}}
// CHECK-LAYOUT-NEXT:    0 |   char c
// CHECK-LAYOUT-NEXT:   16 |   fp80_t x
// CHECK-LAYOUT-NEXT:      | [sizeof=32, align=16,
// CHECK-LAYOUT-NEXT:      |  nvsize=32, nvalign=16]

int typedef_sizes[] = {
    sizeof(TypedefOverAligned), sizeof(TypedefOverAlignedPacked),
    sizeof(TypedefUnderAlignedPacked), sizeof(PlainFp80Packed),
    sizeof(TypedefNoAttrPacked)};
