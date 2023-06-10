// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify -mvscale-min=1 -mvscale-max=1 %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify -mvscale-min=2 -mvscale-max=2 %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify -mvscale-min=4 -mvscale-max=4 %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify -mvscale-min=8 -mvscale-max=8 %s
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +f -target-feature +d -target-feature +zve64d -ffreestanding -fsyntax-only -verify -mvscale-min=16 -mvscale-max=16 %s

#include <stdint.h>

typedef __rvv_bool64_t vbool64_t;
typedef __rvv_bool32_t vbool32_t;
typedef __rvv_bool16_t vbool16_t;
typedef __rvv_bool8_t vbool8_t;
typedef __rvv_bool4_t vbool4_t;
typedef __rvv_bool2_t vbool2_t;
typedef __rvv_bool1_t vbool1_t;

typedef __rvv_int8mf8_t vint8mf8_t;
typedef __rvv_uint8mf8_t vuint8mf8_t;

typedef __rvv_int8mf4_t vint8mf4_t;
typedef __rvv_uint8mf4_t vuint8mf4_t;
typedef __rvv_int16mf4_t vint16mf4_t;
typedef __rvv_uint16mf4_t vuint16mf4_t;

typedef __rvv_int8mf2_t vint8mf2_t;
typedef __rvv_uint8mf2_t vuint8mf2_t;
typedef __rvv_int16mf2_t vint16mf2_t;
typedef __rvv_uint16mf2_t vuint16mf2_t;
typedef __rvv_int32mf2_t vint32mf2_t;
typedef __rvv_uint32mf2_t vuint32mf2_t;
typedef __rvv_float32mf2_t vfloat32mf2_t;

typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_uint8m1_t vuint8m1_t;
typedef __rvv_int16m1_t vint16m1_t;
typedef __rvv_uint16m1_t vuint16m1_t;
typedef __rvv_int32m1_t vint32m1_t;
typedef __rvv_uint32m1_t vuint32m1_t;
typedef __rvv_int64m1_t vint64m1_t;
typedef __rvv_uint64m1_t vuint64m1_t;
typedef __rvv_float32m1_t vfloat32m1_t;
typedef __rvv_float64m1_t vfloat64m1_t;

typedef __rvv_int8m2_t vint8m2_t;
typedef __rvv_uint8m2_t vuint8m2_t;
typedef __rvv_int16m2_t vint16m2_t;
typedef __rvv_uint16m2_t vuint16m2_t;
typedef __rvv_int32m2_t vint32m2_t;
typedef __rvv_uint32m2_t vuint32m2_t;
typedef __rvv_int64m2_t vint64m2_t;
typedef __rvv_uint64m2_t vuint64m2_t;
typedef __rvv_float32m2_t vfloat32m2_t;
typedef __rvv_float64m2_t vfloat64m2_t;

typedef __rvv_int8m4_t vint8m4_t;
typedef __rvv_uint8m4_t vuint8m4_t;
typedef __rvv_int16m4_t vint16m4_t;
typedef __rvv_uint16m4_t vuint16m4_t;
typedef __rvv_int32m4_t vint32m4_t;
typedef __rvv_uint32m4_t vuint32m4_t;
typedef __rvv_int64m4_t vint64m4_t;
typedef __rvv_uint64m4_t vuint64m4_t;
typedef __rvv_float32m4_t vfloat32m4_t;
typedef __rvv_float64m4_t vfloat64m4_t;

typedef __rvv_int8m8_t vint8m8_t;
typedef __rvv_uint8m8_t vuint8m8_t;
typedef __rvv_int16m8_t vint16m8_t;
typedef __rvv_uint16m8_t vuint16m8_t;
typedef __rvv_int32m8_t vint32m8_t;
typedef __rvv_uint32m8_t vuint32m8_t;
typedef __rvv_int64m8_t vint64m8_t;
typedef __rvv_uint64m8_t vuint64m8_t;
typedef __rvv_float32m8_t vfloat32m8_t;
typedef __rvv_float64m8_t vfloat64m8_t;

// Define valid fixed-width RVV types
typedef vint8m1_t fixed_int8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint16m1_t fixed_int16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_int32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint64m1_t fixed_int64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vuint8m1_t fixed_uint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint16m1_t fixed_uint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint32m1_t fixed_uint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint64m1_t fixed_uint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

typedef vfloat32m1_t fixed_float32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t fixed_float64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

// GNU vector types
typedef int8_t gnu_int8m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef int16_t gnu_int16m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef int32_t gnu_int32m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef int64_t gnu_int64m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));

typedef uint8_t gnu_uint8m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef uint16_t gnu_uint16m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef uint32_t gnu_uint32m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef uint64_t gnu_uint64m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));

typedef float gnu_float32m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));
typedef double gnu_float64m1_t __attribute__((vector_size(__riscv_v_fixed_vlen / 8)));

typedef int32_t gnu_int32m2_t __attribute__((vector_size((__riscv_v_fixed_vlen * 2) / 8)));

// Attribute must have a single argument
typedef vint8m1_t no_argument __attribute__((riscv_rvv_vector_bits));         // expected-error {{'riscv_rvv_vector_bits' attribute takes one argument}}
typedef vint8m1_t two_arguments __attribute__((riscv_rvv_vector_bits(2, 4))); // expected-error {{'riscv_rvv_vector_bits' attribute takes one argument}}

// The number of RVV vector bits must be an integer constant expression
typedef vint8m1_t non_int_size1 __attribute__((riscv_rvv_vector_bits(2.0)));   // expected-error {{'riscv_rvv_vector_bits' attribute requires an integer constant}}
typedef vint8m1_t non_int_size2 __attribute__((riscv_rvv_vector_bits("256"))); // expected-error {{'riscv_rvv_vector_bits' attribute requires an integer constant}}

// bool types and LMUL != 1 are not supported.
typedef vbool1_t fixed_vbool1_t_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vbool1_t'}}
typedef vint8mf8_t fixed_int8mf8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 8))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8mf8_t'}}
typedef vint8mf4_t fixed_int8mf4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 4))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8mf4_t'}}
typedef vint8mf2_t fixed_int8mf2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen / 2))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8mf2_t'}}
typedef vint8m2_t fixed_int8m2_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 2))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8m2_t'}}
typedef vint8m4_t fixed_int8m4_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 4))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8m4_t'}}
typedef vint8m8_t fixed_int8m8_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen * 8))); // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vint8m8_t'}}

// Attribute must be attached to a single RVV vector or predicate type.
typedef void *badtype1 __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));         // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'void *'}}
typedef int badtype2 __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));           // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'int'}}
typedef float badtype3 __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));         // expected-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'float'}}

// Attribute only applies to typedefs.
vint8m1_t non_typedef_type __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));  // expected-error {{'riscv_rvv_vector_bits' attribute only applies to typedefs}}

// Test that we can define non-local fixed-length RVV types (unsupported for
// sizeless types).
fixed_int8m1_t global_int8;

extern fixed_int8m1_t extern_int8;

static fixed_int8m1_t static_int8;

fixed_int8m1_t *global_int8_ptr;
extern fixed_int8m1_t *extern_int8_ptr;
static fixed_int8m1_t *static_int8_ptr;
__thread fixed_int8m1_t thread_int8;

typedef fixed_int8m1_t int8_typedef;
typedef fixed_int8m1_t *int8_ptr_typedef;

// Test sized expressions
int sizeof_int8 = sizeof(global_int8);
int sizeof_int8_var = sizeof(*global_int8_ptr);
int sizeof_int8_var_ptr = sizeof(global_int8_ptr);

extern fixed_int8m1_t *extern_int8_ptr;

int alignof_int8 = __alignof__(extern_int8);
int alignof_int8_var = __alignof__(*extern_int8_ptr);
int alignof_int8_var_ptr = __alignof__(extern_int8_ptr);

void f(int c) {
  fixed_int8m1_t fs8;
  vint8m1_t ss8;
  gnu_int8m1_t gs8;

  // Check conditional expressions where the result is ambiguous are
  // ill-formed.
  void *sel __attribute__((unused));
  sel = c ? ss8 : fs8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  sel = c ? fs8 : ss8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}

  sel = c ? gs8 : ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  sel = c ? ss8 : gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  sel = c ? gs8 : fs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  sel = c ? fs8 : gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  // Check binary expressions where the result is ambiguous are ill-formed.
  ss8 = ss8 + fs8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  ss8 = ss8 + gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  fs8 = fs8 + ss8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  fs8 = fs8 + gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 = gs8 + ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  gs8 = gs8 + fs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 += fs8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  ss8 += gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  fs8 += ss8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  fs8 += gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 += ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  gs8 += fs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 = ss8 == fs8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  ss8 = ss8 == gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  fs8 = fs8 == ss8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  fs8 = fs8 == gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 = gs8 == ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  gs8 = gs8 == fs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  ss8 = ss8 & fs8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  ss8 = ss8 & gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  fs8 = fs8 & ss8; // expected-error {{cannot combine fixed-length and sizeless RVV vectors in expression, result is ambiguous}}
  fs8 = fs8 & gs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}

  gs8 = gs8 & ss8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
  gs8 = gs8 & fs8; // expected-error {{cannot combine GNU and RVV vectors in expression, result is ambiguous}}
}

// --------------------------------------------------------------------------//
// Sizeof

#define VECTOR_SIZE ((__riscv_v_fixed_vlen / 8))

_Static_assert(sizeof(fixed_int8m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int16m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int32m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int64m1_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_uint8m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_uint16m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_uint32m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int64m1_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_float32m1_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_float64m1_t) == VECTOR_SIZE, "");

// --------------------------------------------------------------------------//
// Alignof

#define VECTOR_ALIGN 8

_Static_assert(__alignof__(fixed_int8m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int16m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int32m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int64m1_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_uint8m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint16m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint32m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint64m1_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_float32m1_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_float64m1_t) == VECTOR_ALIGN, "");

// --------------------------------------------------------------------------//
// Structs

struct struct_int64 { fixed_int64m1_t x, y[5]; };
struct struct_float64 { fixed_float64m1_t x, y[5]; };

// --------------------------------------------------------------------------//
// Unions
union union_int64 { fixed_int64m1_t x, y[5]; };
union union_float64 { fixed_float64m1_t x, y[5]; };

// --------------------------------------------------------------------------//
// Implicit casts

#define TEST_CAST_COMMON(TYPE)                                              \
  v##TYPE##_t to_v##TYPE##_t_from_fixed(fixed_##TYPE##_t x) { return x; } \
  fixed_##TYPE##_t from_##TYPE##_t_to_fixed(v##TYPE##_t x) { return x; }

#define TEST_CAST_GNU(PREFIX, TYPE)                                                          \
  gnu_##TYPE##_t to_gnu_##TYPE##_t_from_##PREFIX##TYPE##_t(PREFIX##TYPE##_t x) { return x; } \
  PREFIX##TYPE##_t from_gnu_##TYPE##_t_to_##PREFIX##TYPE##_t(gnu_##TYPE##_t x) { return x; }

#define TEST_CAST_VECTOR(TYPE) \
  TEST_CAST_COMMON(TYPE)       \
  TEST_CAST_GNU(v, TYPE)      \
  TEST_CAST_GNU(fixed_, TYPE)

TEST_CAST_VECTOR(int8m1)
TEST_CAST_VECTOR(int16m1)
TEST_CAST_VECTOR(int32m1)
TEST_CAST_VECTOR(int64m1)
TEST_CAST_VECTOR(uint8m1)
TEST_CAST_VECTOR(uint16m1)
TEST_CAST_VECTOR(uint32m1)
TEST_CAST_VECTOR(uint64m1)
TEST_CAST_VECTOR(float32m1)
TEST_CAST_VECTOR(float64m1)

// Test that casts only work for LMUL=1 types and don't crash.
vint32m2_t to_vint32m2_t_from_gnut(gnu_int32m2_t x) { return x; } // expected-error-re {{returning 'gnu_int32m2_t' (vector of {{[0-9]+}} 'int32_t' values) from a function with incompatible result type 'vint32m2_t' (aka '__rvv_int32m2_t')}}

gnu_int32m2_t to_gnut_from_svint32_t(vint32m2_t x) { return x; } // expected-error-re {{returning 'vint32m2_t' (aka '__rvv_int32m2_t') from a function with incompatible result type 'gnu_int32m2_t' (vector of {{[0-9]+}} 'int32_t' values)}}

// --------------------------------------------------------------------------//
// Test the scalable and fixed-length types can be used interchangeably

vint32m1_t __attribute__((overloadable)) vfunc(vint32m1_t op1, vint32m1_t op2);
vfloat64m1_t __attribute__((overloadable)) vfunc(vfloat64m1_t op1, vfloat64m1_t op2);

#define TEST_CALL(TYPE)                                              \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_ff(fixed_##TYPE##_t op1, fixed_##TYPE##_t op2) { \
    return vfunc(op1, op2);                                         \
  }                                                                  \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_fs(fixed_##TYPE##_t op1, v##TYPE##_t op2) {     \
    return vfunc(op1, op2);                                         \
  }                                                                  \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_sf(v##TYPE##_t op1, fixed_##TYPE##_t op2) {     \
    return vfunc(op1, op2);                                         \
  }

TEST_CALL(int32m1)
TEST_CALL(float64m1)

// --------------------------------------------------------------------------//
// Vector initialization

#if __riscv_v_fixed_vlen == 256

typedef vint32m1_t int32x8 __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t float64x4 __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

int32x8 foo = {1, 2, 3, 4, 5, 6, 7, 8};
int32x8 foo2 = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // expected-warning{{excess elements in vector initializer}}

float64x4 bar = {1.0, 2.0, 3.0, 4.0};
float64x4 bar2 = {1.0, 2.0, 3.0, 4.0, 5.0}; // expected-warning{{excess elements in vector initializer}}

#endif

// --------------------------------------------------------------------------//
// Vector ops

#define TEST_BINARY(TYPE, NAME, OP)                  \
  TYPE NAME##_##TYPE(TYPE op1, TYPE op2) {           \
    return op1 OP op2;                               \
  }                                                  \
  TYPE compound##NAME##_##TYPE(TYPE op1, TYPE op2) { \
    op1 OP##= op2;                                   \
    return op1;                                      \
  }

#define TEST_COMPARISON(TYPE, NAME, OP)    \
  TYPE NAME##_##TYPE(TYPE op1, TYPE op2) { \
    return op1 OP op2;                     \
  }

#define TEST_UNARY(TYPE, NAME, OP) \
  TYPE NAME##_##TYPE(TYPE op1) {   \
    return OP op1;                 \
  }

#define TEST_OPS(TYPE)           \
  TEST_BINARY(TYPE, add, +)      \
  TEST_BINARY(TYPE, sub, -)      \
  TEST_BINARY(TYPE, mul, *)      \
  TEST_BINARY(TYPE, div, /)      \
  TEST_COMPARISON(TYPE, eq, ==)  \
  TEST_COMPARISON(TYPE, ne, !=)  \
  TEST_COMPARISON(TYPE, lt, <)   \
  TEST_COMPARISON(TYPE, gt, >)   \
  TEST_COMPARISON(TYPE, lte, <=) \
  TEST_COMPARISON(TYPE, gte, >=) \
  TEST_UNARY(TYPE, nop, +)       \
  TEST_UNARY(TYPE, neg, -)

#define TEST_INT_OPS(TYPE)   \
  TEST_OPS(TYPE)             \
  TEST_BINARY(TYPE, mod, %)  \
  TEST_BINARY(TYPE, and, &)  \
  TEST_BINARY(TYPE, or, |)   \
  TEST_BINARY(TYPE, xor, ^)  \
  TEST_BINARY(TYPE, shl, <<) \
  TEST_BINARY(TYPE, shr, <<) \
  TEST_UNARY(TYPE, not, ~)

TEST_INT_OPS(fixed_int8m1_t)
TEST_INT_OPS(fixed_int16m1_t)
TEST_INT_OPS(fixed_int32m1_t)
TEST_INT_OPS(fixed_int64m1_t)
TEST_INT_OPS(fixed_uint8m1_t)
TEST_INT_OPS(fixed_uint16m1_t)
TEST_INT_OPS(fixed_uint32m1_t)
TEST_INT_OPS(fixed_uint64m1_t)

TEST_OPS(fixed_float32m1_t)
TEST_OPS(fixed_float64m1_t)
