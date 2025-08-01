// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -target-feature +sme -ffreestanding -fsyntax-only -verify=expected,streamingdifferent -mvscale-min=1 -mvscale-max=1 %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -target-feature +sme -ffreestanding -fsyntax-only -verify=expected,streamingdifferent -mvscale-min=2 -mvscale-max=2 -mvscale-streaming-min=2 %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -target-feature +sme -ffreestanding -fsyntax-only -verify=expected -mvscale-min=4 -mvscale-max=4 -mvscale-streaming-min=4 -mvscale-streaming-max=4 %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -target-feature +sme -ffreestanding -fsyntax-only -verify=expected,streamingdifferent -mvscale-min=8 -mvscale-max=8 -mvscale-streaming-min=4 -mvscale-streaming-max=8 %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -target-feature +sme -ffreestanding -fsyntax-only -verify=expected,streamingdifferent -mvscale-min=16 -mvscale-max=16 %s

#include <stdint.h>

#define N __ARM_FEATURE_SVE_BITS

typedef __fp16 float16_t;
typedef float float32_t;
typedef double float64_t;
typedef __SVInt8_t svint8_t;
typedef __SVInt16_t svint16_t;
typedef __SVInt32_t svint32_t;
typedef __SVInt64_t svint64_t;
typedef __SVUint8_t svuint8_t;
typedef __SVUint16_t svuint16_t;
typedef __SVUint32_t svuint32_t;
typedef __SVUint64_t svuint64_t;
typedef __SVFloat16_t svfloat16_t;
typedef __SVFloat32_t svfloat32_t;
typedef __SVFloat64_t svfloat64_t;

#if defined(__ARM_FEATURE_SVE_BF16)
typedef __bf16 bfloat16_t;
typedef __SVBfloat16_t svbfloat16_t;
#endif

typedef __SVBool_t svbool_t;

// Define valid fixed-width SVE types
typedef svint8_t fixed_int8_t __attribute__((arm_sve_vector_bits(N)));
typedef svint16_t fixed_int16_t __attribute__((arm_sve_vector_bits(N)));
typedef svint32_t fixed_int32_t __attribute__((arm_sve_vector_bits(N)));
typedef svint64_t fixed_int64_t __attribute__((arm_sve_vector_bits(N)));

typedef svuint8_t fixed_uint8_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint16_t fixed_uint16_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint32_t fixed_uint32_t __attribute__((arm_sve_vector_bits(N)));
typedef svuint64_t fixed_uint64_t __attribute__((arm_sve_vector_bits(N)));

typedef svfloat16_t fixed_float16_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat32_t fixed_float32_t __attribute__((arm_sve_vector_bits(N)));
typedef svfloat64_t fixed_float64_t __attribute__((arm_sve_vector_bits(N)));

typedef svbfloat16_t fixed_bfloat16_t __attribute__((arm_sve_vector_bits(N)));

typedef svbool_t fixed_bool_t __attribute__((arm_sve_vector_bits(N)));

// GNU vector types
typedef int8_t gnu_int8_t __attribute__((vector_size(N / 8)));
typedef int16_t gnu_int16_t __attribute__((vector_size(N / 8)));
typedef int32_t gnu_int32_t __attribute__((vector_size(N / 8)));
typedef int64_t gnu_int64_t __attribute__((vector_size(N / 8)));

typedef uint8_t gnu_uint8_t __attribute__((vector_size(N / 8)));
typedef uint16_t gnu_uint16_t __attribute__((vector_size(N / 8)));
typedef uint32_t gnu_uint32_t __attribute__((vector_size(N / 8)));
typedef uint64_t gnu_uint64_t __attribute__((vector_size(N / 8)));

typedef float16_t gnu_float16_t __attribute__((vector_size(N / 8)));
typedef float32_t gnu_float32_t __attribute__((vector_size(N / 8)));
typedef float64_t gnu_float64_t __attribute__((vector_size(N / 8)));

typedef bfloat16_t gnu_bfloat16_t __attribute__((vector_size(N / 8)));

// Attribute must have a single argument
typedef svint8_t no_argument __attribute__((arm_sve_vector_bits));         // expected-error {{'arm_sve_vector_bits' attribute takes one argument}}
typedef svint8_t two_arguments __attribute__((arm_sve_vector_bits(2, 4))); // expected-error {{'arm_sve_vector_bits' attribute takes one argument}}

// The number of SVE vector bits must be an integer constant expression
typedef svint8_t non_int_size1 __attribute__((arm_sve_vector_bits(2.0)));   // expected-error {{'arm_sve_vector_bits' attribute requires an integer constant}}
typedef svint8_t non_int_size2 __attribute__((arm_sve_vector_bits("256"))); // expected-error {{'arm_sve_vector_bits' attribute requires an integer constant}}

typedef __clang_svint8x2_t svint8x2_t;
typedef __clang_svfloat32x3_t svfloat32x3_t;

// Attribute must be attached to a single SVE vector or predicate type.
typedef void *badtype1 __attribute__((arm_sve_vector_bits(N)));         // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'void *'}}
typedef int badtype2 __attribute__((arm_sve_vector_bits(N)));           // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'int'}}
typedef float badtype3 __attribute__((arm_sve_vector_bits(N)));         // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'float'}}
typedef svint8x2_t badtype4 __attribute__((arm_sve_vector_bits(N)));    // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'svint8x2_t' (aka '__clang_svint8x2_t')}}
typedef svfloat32x3_t badtype5 __attribute__((arm_sve_vector_bits(N))); // expected-error {{'arm_sve_vector_bits' attribute applied to non-SVE type 'svfloat32x3_t' (aka '__clang_svfloat32x3_t')}}

// Attribute only applies to typedefs.
svint8_t non_typedef_type __attribute__((arm_sve_vector_bits(N)));  // expected-error {{'arm_sve_vector_bits' attribute only applies to typedefs}}

// Test that we can define non-local fixed-length SVE types (unsupported for
// sizeless types).
fixed_int8_t global_int8;
fixed_bfloat16_t global_bfloat16;
fixed_bool_t global_bool;

extern fixed_int8_t extern_int8;
extern fixed_bfloat16_t extern_bfloat16;
extern fixed_bool_t extern_bool;

static fixed_int8_t static_int8;
static fixed_bfloat16_t static_bfloat16;
static fixed_bool_t static_bool;

fixed_int8_t *global_int8_ptr;
extern fixed_int8_t *extern_int8_ptr;
static fixed_int8_t *static_int8_ptr;
__thread fixed_int8_t thread_int8;

typedef fixed_int8_t int8_typedef;
typedef fixed_int8_t *int8_ptr_typedef;

// Test sized expressions
int sizeof_int8 = sizeof(global_int8);
int sizeof_int8_var = sizeof(*global_int8_ptr);
int sizeof_int8_var_ptr = sizeof(global_int8_ptr);

extern fixed_int8_t *extern_int8_ptr;

int alignof_int8 = __alignof__(extern_int8);
int alignof_int8_var = __alignof__(*extern_int8_ptr);
int alignof_int8_var_ptr = __alignof__(extern_int8_ptr);

void f(int c) {
  fixed_int8_t fs8;
  svint8_t ss8;
  gnu_int8_t gs8;

  // Check conditional expressions where the result is ambiguous are
  // ill-formed.
  void *sel __attribute__((unused));
  sel = c ? ss8 : fs8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  sel = c ? fs8 : ss8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}

  sel = c ? gs8 : ss8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  sel = c ? ss8 : gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  sel = c ? gs8 : fs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  sel = c ? fs8 : gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  // Check binary expressions where the result is ambiguous are ill-formed.
  ss8 = ss8 + fs8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  ss8 = ss8 + gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  fs8 = fs8 + ss8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  fs8 = fs8 + gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  gs8 = gs8 + ss8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  gs8 = gs8 + fs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  ss8 += fs8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  ss8 += gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  fs8 += ss8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  fs8 += gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  gs8 += ss8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  gs8 += fs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  ss8 = ss8 == fs8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  ss8 = ss8 == gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  fs8 = fs8 == ss8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  fs8 = fs8 == gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  gs8 = gs8 == ss8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  gs8 = gs8 == fs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  ss8 = ss8 & fs8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  ss8 = ss8 & gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  fs8 = fs8 & ss8; // expected-error {{cannot combine fixed-length and sizeless SVE vectors in expression, result is ambiguous}}
  fs8 = fs8 & gs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}

  gs8 = gs8 & ss8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
  gs8 = gs8 & fs8; // expected-error {{cannot combine GNU and SVE vectors in expression, result is ambiguous}}
}

// --------------------------------------------------------------------------//
// Sizeof

#define VECTOR_SIZE ((N / 8))
#define PRED_SIZE ((N / 64))

_Static_assert(sizeof(fixed_int8_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_int16_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int32_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_int64_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_uint8_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_uint16_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_uint32_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_uint64_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_float16_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_float32_t) == VECTOR_SIZE, "");
_Static_assert(sizeof(fixed_float64_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_bfloat16_t) == VECTOR_SIZE, "");

_Static_assert(sizeof(fixed_bool_t) == PRED_SIZE, "");

// --------------------------------------------------------------------------//
// Alignof

#define VECTOR_ALIGN 16
#define PRED_ALIGN 2

_Static_assert(__alignof__(fixed_int8_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int16_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int32_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_int64_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_uint8_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint16_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint32_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_uint64_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_float16_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_float32_t) == VECTOR_ALIGN, "");
_Static_assert(__alignof__(fixed_float64_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_bfloat16_t) == VECTOR_ALIGN, "");

_Static_assert(__alignof__(fixed_bool_t) == PRED_ALIGN, "");

// --------------------------------------------------------------------------//
// Structs

struct struct_int64 { fixed_int64_t x, y[5]; };
struct struct_float64 { fixed_float64_t x, y[5]; };
struct struct_bfloat16 { fixed_bfloat16_t x, y[5]; };
struct struct_bool { fixed_bool_t x, y[5]; };

// --------------------------------------------------------------------------//
// Unions
union union_int64 { fixed_int64_t x, y[5]; };
union union_float64 { fixed_float64_t x, y[5]; };
union union_bfloat16 { fixed_bfloat16_t x, y[5]; };
union union_bool { fixed_bool_t x, y[5]; };

// --------------------------------------------------------------------------//
// Implicit casts

#define TEST_CAST_COMMON(TYPE)                                              \
  sv##TYPE##_t to_sv##TYPE##_t_from_fixed(fixed_##TYPE##_t x) { return x; } \
  fixed_##TYPE##_t from_sv##TYPE##_t_to_fixed(sv##TYPE##_t x) { return x; }

#define TEST_CAST_GNU(PREFIX, TYPE)                                                          \
  gnu_##TYPE##_t to_gnu_##TYPE##_t_from_##PREFIX##TYPE##_t(PREFIX##TYPE##_t x) { return x; } \
  PREFIX##TYPE##_t from_gnu_##TYPE##_t_to_##PREFIX##TYPE##_t(gnu_##TYPE##_t x) { return x; }

#define TEST_CAST_VECTOR(TYPE) \
  TEST_CAST_COMMON(TYPE)       \
  TEST_CAST_GNU(sv, TYPE)      \
  TEST_CAST_GNU(fixed_, TYPE)

TEST_CAST_VECTOR(int8)
TEST_CAST_VECTOR(int16)
TEST_CAST_VECTOR(int32)
TEST_CAST_VECTOR(int64)
TEST_CAST_VECTOR(uint8)
TEST_CAST_VECTOR(uint16)
TEST_CAST_VECTOR(uint32)
TEST_CAST_VECTOR(uint64)
TEST_CAST_VECTOR(float16)
TEST_CAST_VECTOR(float32)
TEST_CAST_VECTOR(float64)
TEST_CAST_VECTOR(bfloat16)
TEST_CAST_COMMON(bool)

// Test the implicit conversion only applies to valid types
fixed_bool_t to_fixed_bool_t__from_svint32_t(svint32_t x) { return x; } // expected-error-re {{returning 'svint32_t' (aka '__SVInt32_t') from a function with incompatible result type 'fixed_bool_t' (vector of {{[0-9]+}} 'unsigned char' values)}}

// Test implicit conversion between SVE and GNU vector is invalid when
// __ARM_FEATURE_SVE_BITS != N
#if defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS == 512
typedef int32_t int4 __attribute__((vector_size(16)));
svint32_t badcast(int4 x) { return x; } // expected-error {{returning 'int4' (vector of 4 'int32_t' values) from a function with incompatible result type 'svint32_t' (aka '__SVInt32_t')}}
#endif

// Test conversion between predicate and uint8 is invalid, both have the same
// memory representation.
fixed_bool_t to_fixed_bool_t__from_svuint8_t(svuint8_t x) { return x; } // expected-error-re {{returning 'svuint8_t' (aka '__SVUint8_t') from a function with incompatible result type 'fixed_bool_t' (vector of {{[0-9]+}} 'unsigned char' values)}}

// --------------------------------------------------------------------------//
// Test the scalable and fixed-length types can be used interchangeably

svint32_t __attribute__((overloadable)) svfunc(svint32_t op1, svint32_t op2);
svfloat64_t __attribute__((overloadable)) svfunc(svfloat64_t op1, svfloat64_t op2);
svbool_t __attribute__((overloadable)) svfunc(svbool_t op1, svbool_t op2);

#define TEST_CALL(TYPE)                                              \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_ff(fixed_##TYPE##_t op1, fixed_##TYPE##_t op2) { \
    return svfunc(op1, op2);                                         \
  }                                                                  \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_fs(fixed_##TYPE##_t op1, sv##TYPE##_t op2) {     \
    return svfunc(op1, op2);                                         \
  }                                                                  \
  fixed_##TYPE##_t                                                   \
      call_##TYPE##_sf(sv##TYPE##_t op1, fixed_##TYPE##_t op2) {     \
    return svfunc(op1, op2);                                         \
  }

TEST_CALL(int32)
TEST_CALL(float64)
TEST_CALL(bool)

// --------------------------------------------------------------------------//
// Vector initialization

#if __ARM_FEATURE_SVE_BITS == 256

typedef svint32_t int32x8 __attribute__((arm_sve_vector_bits(N)));
typedef svfloat64_t float64x4 __attribute__((arm_sve_vector_bits(N)));

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

TEST_INT_OPS(fixed_int8_t)
TEST_INT_OPS(fixed_int16_t)
TEST_INT_OPS(fixed_int32_t)
TEST_INT_OPS(fixed_int64_t)
TEST_INT_OPS(fixed_uint8_t)
TEST_INT_OPS(fixed_uint16_t)
TEST_INT_OPS(fixed_uint32_t)
TEST_INT_OPS(fixed_uint64_t)

TEST_OPS(fixed_float16_t)
TEST_OPS(fixed_float32_t)
TEST_OPS(fixed_float64_t)

// --------------------------------------------------------------------------//
// Streaming
__arm_locally_streaming void locally_streaming() {
  svint8_t t1 = extern_int8; // streamingdifferent-error {{cannot be used in a streaming function}}
  svbool_t t2 = extern_bool; // streamingdifferent-error {{cannot be used in a streaming function}}
  void* t3 = extern_int8_ptr;
}
void streaming(void) __arm_streaming {
  svint8_t t1 = extern_int8; // streamingdifferent-error {{cannot be used in a streaming function}}
  svbool_t t2 = extern_bool; // streamingdifferent-error {{cannot be used in a streaming function}}
  void* t3 = extern_int8_ptr;
}
void streaming_compatible(void) __arm_streaming_compatible {
  svint8_t t1 = extern_int8; // streamingdifferent-error {{cannot be used in a streaming-compatible function}} \
                             // streamingdifferent-error {{initializing}}
  svbool_t t2 = extern_bool; // streamingdifferent-error {{cannot be used in a streaming-compatible function}} \
                             // streamingdifferent-error {{initializing}}
  void* t3 = extern_int8_ptr;
}
__arm_locally_streaming void locally_streaming_arg(fixed_int8_t x) {} // streamingdifferent-error {{cannot be used in a streaming function}}
