// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=128 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=256 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=512 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=1024 -fallow-half-arguments-and-returns %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fsyntax-only -verify -msve-vector-bits=2048 -fallow-half-arguments-and-returns %s

#define N __ARM_FEATURE_SVE_BITS_EXPERIMENTAL

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
typedef __SVBFloat16_t svbfloat16_t;
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

  void *sel __attribute__((unused));
  sel = c ? ss8 : fs8; // expected-error {{cannot convert between a fixed-length and a sizeless vector}}
  sel = c ? fs8 : ss8; // expected-error {{cannot convert between a fixed-length and a sizeless vector}}

  sel = fs8 + ss8; // expected-error {{cannot convert between a fixed-length and a sizeless vector}}
  sel = ss8 + fs8; // expected-error {{cannot convert between a fixed-length and a sizeless vector}}
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

#define TEST_CAST(TYPE)                                          \
  sv##TYPE##_t to_sv##TYPE##_t(fixed_##TYPE##_t x) { return x; } \
  fixed_##TYPE##_t from_sv##TYPE##_t(sv##TYPE##_t x) { return x; }

TEST_CAST(int8)
TEST_CAST(int16)
TEST_CAST(int32)
TEST_CAST(int64)
TEST_CAST(uint8)
TEST_CAST(uint16)
TEST_CAST(uint32)
TEST_CAST(uint64)
TEST_CAST(float16)
TEST_CAST(float32)
TEST_CAST(float64)
TEST_CAST(bfloat16)
TEST_CAST(bool)

// Test the implicit conversion only applies to valid types
fixed_int8_t to_fixed_int8_t__from_svuint8_t(svuint8_t x) { return x; } // expected-error-re {{returning 'svuint8_t' (aka '__SVUint8_t') from a function with incompatible result type 'fixed_int8_t' (vector of {{[0-9]+}} 'signed char' values)}}
fixed_bool_t to_fixed_bool_t__from_svint32_t(svint32_t x) { return x; } // expected-error-re {{returning 'svint32_t' (aka '__SVInt32_t') from a function with incompatible result type 'fixed_bool_t' (vector of {{[0-9]+}} 'unsigned char' values)}}

// Test conversion between predicate and uint8 is invalid, both have the same
// memory representation.
fixed_bool_t to_fixed_bool_t__from_svuint8_t(svuint8_t x) { return x; } // expected-error-re {{returning 'svuint8_t' (aka '__SVUint8_t') from a function with incompatible result type 'fixed_bool_t' (vector of {{[0-9]+}} 'unsigned char' values)}}

// Test the implicit conversion only applies to fixed-length types
typedef signed int vSInt32 __attribute__((__vector_size__(16)));
svint32_t to_svint32_t_from_gnut(vSInt32 x) { return x; } // expected-error-re {{returning 'vSInt32' (vector of {{[0-9]+}} 'int' values) from a function with incompatible result type 'svint32_t' (aka '__SVInt32_t')}}

vSInt32 to_gnut_from_svint32_t(svint32_t x) { return x; } // expected-error-re {{returning 'svint32_t' (aka '__SVInt32_t') from a function with incompatible result type 'vSInt32' (vector of {{[0-9]+}} 'int' values)}}

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
