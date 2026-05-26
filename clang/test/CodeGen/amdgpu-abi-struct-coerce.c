// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s | FileCheck %s

// Test AMDGPU ABI struct coercion behavior:
// - Structs containing ONLY sub-32-bit integers (char, short) should be packed into i32 registers
// - Structs containing floats or full-sized integers (i32, i64) should preserve their original types
//
// This tests the fix for the issue where structs like {float, int} were incorrectly
// coerced to [2 x i32], losing float type information.

// ============================================================================
// SECTION 1: Structs with floats - should NOT be coerced to integers
// ============================================================================

typedef struct fp_int_pair {
    float f;
    int i;
} fp_int_pair;

// CHECK-LABEL: define{{.*}} %struct.fp_int_pair @return_fp_int_pair(float %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.fp_int_pair
fp_int_pair return_fp_int_pair(fp_int_pair x) {
    return x;
}

typedef struct int_fp_pair {
    int i;
    float f;
} int_fp_pair;

// CHECK-LABEL: define{{.*}} %struct.int_fp_pair @return_int_fp_pair(i32 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.int_fp_pair
int_fp_pair return_int_fp_pair(int_fp_pair x) {
    return x;
}

typedef struct two_floats {
    float a;
    float b;
} two_floats;

// Two floats can be packed into <2 x float> vector
// CHECK-LABEL: define{{.*}} <2 x float> @return_two_floats(<2 x float> %x.coerce)
two_floats return_two_floats(two_floats x) {
    return x;
}

// Double precision floats
typedef struct double_struct {
    double d;
} double_struct;

// CHECK-LABEL: define{{.*}} double @return_double_struct(double %x.coerce)
double_struct return_double_struct(double_struct x) {
    return x;
}

// ============================================================================
// SECTION 2: Structs with full-sized integers - should NOT be coerced
// ============================================================================

typedef struct two_ints {
    int a;
    int b;
} two_ints;

// CHECK-LABEL: define{{.*}} %struct.two_ints @return_two_ints(i32 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.two_ints
two_ints return_two_ints(two_ints x) {
    return x;
}

typedef struct single_int {
    int a;
} single_int;

// CHECK-LABEL: define{{.*}} i32 @return_single_int(i32 %x.coerce)
single_int return_single_int(single_int x) {
    return x;
}

typedef struct int64_struct {
    long long a;
} int64_struct;

// CHECK-LABEL: define{{.*}} i64 @return_int64_struct(i64 %x.coerce)
int64_struct return_int64_struct(int64_struct x) {
    return x;
}

// ============================================================================
// SECTION 3: Structs with ONLY sub-32-bit integers - SHOULD be coerced
// ============================================================================

// Structs of small integers <= 32 bits should be coerced to i32
typedef struct small_struct {
    short a;
    short b;
} small_struct;

// CHECK-LABEL: define{{.*}} i32 @return_small_struct(i32 %x.coerce)
small_struct return_small_struct(small_struct x) {
    return x;
}

// Structs of small integers <= 16 bits should be coerced to i16
typedef struct tiny_struct {
    char a;
    char b;
} tiny_struct;

// CHECK-LABEL: define{{.*}} i16 @return_tiny_struct(i16 %x.coerce)
tiny_struct return_tiny_struct(tiny_struct x) {
    return x;
}

// Struct of 8 chars (64 bits) should be coerced to [2 x i32]
typedef struct eight_chars {
    char a, b, c, d, e, f, g, h;
} eight_chars;

// CHECK-LABEL: define{{.*}} [2 x i32] @return_eight_chars([2 x i32] %x.coerce)
eight_chars return_eight_chars(eight_chars x) {
    return x;
}

// Struct of 4 chars (32 bits) should be coerced to i32
typedef struct four_chars {
    char a, b, c, d;
} four_chars;

// CHECK-LABEL: define{{.*}} i32 @return_four_chars(i32 %x.coerce)
four_chars return_four_chars(four_chars x) {
    return x;
}

// Struct of 4 shorts (64 bits) should be coerced to [2 x i32]
typedef struct four_shorts {
    short a, b, c, d;
} four_shorts;

// CHECK-LABEL: define{{.*}} [2 x i32] @return_four_shorts([2 x i32] %x.coerce)
four_shorts return_four_shorts(four_shorts x) {
    return x;
}

// ============================================================================
// SECTION 4: Mixed types - floats prevent coercion even with small integers
// ============================================================================

typedef struct char_and_float {
    char c;
    float f;
} char_and_float;

// CHECK-LABEL: define{{.*}} %struct.char_and_float @return_char_and_float(i8 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.char_and_float
char_and_float return_char_and_float(char_and_float x) {
    return x;
}

typedef struct short_and_float {
    short s;
    float f;
} short_and_float;

// CHECK-LABEL: define{{.*}} %struct.short_and_float @return_short_and_float(i16 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.short_and_float
short_and_float return_short_and_float(short_and_float x) {
    return x;
}

// Small int + full-sized int should NOT be coerced
typedef struct char_and_int {
    char c;
    int i;
} char_and_int;

// CHECK-LABEL: define{{.*}} %struct.char_and_int @return_char_and_int(i8 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.char_and_int
char_and_int return_char_and_int(char_and_int x) {
    return x;
}

// ============================================================================
// SECTION 5: Exotic/Complex aggregates (per reviewer request)
// ============================================================================

// --- Nested structs ---

typedef struct inner_chars {
    char a, b;
} inner_chars;

typedef struct outer_with_inner_chars {
    inner_chars inner;
    char c, d;
} outer_with_inner_chars;

// All chars, 32 bits total - should be coerced to i32
// CHECK-LABEL: define{{.*}} i32 @return_nested_chars(i32 %x.coerce)
outer_with_inner_chars return_nested_chars(outer_with_inner_chars x) {
    return x;
}

typedef struct inner_with_float {
    char c;
    float f;
} inner_with_float;

typedef struct outer_with_float_inner {
    inner_with_float inner;
} outer_with_float_inner;

// Nested struct contains float - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.outer_with_float_inner @return_nested_with_float(%struct.inner_with_float %x.coerce)
// CHECK: ret %struct.outer_with_float_inner
outer_with_float_inner return_nested_with_float(outer_with_float_inner x) {
    return x;
}

// --- Arrays within structs ---

typedef struct char_array_struct {
    char arr[4];
} char_array_struct;

// Array of 4 chars = 32 bits, all small ints - should be coerced to i32
// CHECK-LABEL: define{{.*}} i32 @return_char_array(i32 %x.coerce)
char_array_struct return_char_array(char_array_struct x) {
    return x;
}

typedef struct short_array_struct {
    short arr[2];
} short_array_struct;

// Array of 2 shorts = 32 bits, all small ints - should be coerced to i32
// CHECK-LABEL: define{{.*}} i32 @return_short_array(i32 %x.coerce)
short_array_struct return_short_array(short_array_struct x) {
    return x;
}

typedef struct int_array_struct {
    int arr[2];
} int_array_struct;

// Array of 2 ints = 64 bits, but ints are full-sized - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.int_array_struct @return_int_array([2 x i32] %x.coerce)
// CHECK: ret %struct.int_array_struct
int_array_struct return_int_array(int_array_struct x) {
    return x;
}

typedef struct float_array_struct {
    float arr[2];
} float_array_struct;

// Array of 2 floats - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.float_array_struct @return_float_array([2 x float] %x.coerce)
// CHECK: ret %struct.float_array_struct
float_array_struct return_float_array(float_array_struct x) {
    return x;
}

// --- Complex combinations ---

typedef struct mixed_nested {
    struct {
        char a;
        char b;
    } inner;
    short s;
} mixed_nested;

// All small integers (nested anonymous struct + short) = 32 bits - should be coerced
// CHECK-LABEL: define{{.*}} i32 @return_mixed_nested(i32 %x.coerce)
mixed_nested return_mixed_nested(mixed_nested x) {
    return x;
}

typedef struct deeply_nested_chars {
    struct {
        struct {
            char a, b;
        } level2;
        char c, d;
    } level1;
} deeply_nested_chars;

// Deeply nested, but all chars = 32 bits - should be coerced
// CHECK-LABEL: define{{.*}} i32 @return_deeply_nested(i32 %x.coerce)
deeply_nested_chars return_deeply_nested(deeply_nested_chars x) {
    return x;
}

typedef struct deeply_nested_with_float {
    struct {
        struct {
            char a;
            float f;  // Float buried deep
        } level2;
    } level1;
} deeply_nested_with_float;

// Float buried in nested struct - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.deeply_nested_with_float @return_deeply_nested_float
// CHECK: ret %struct.deeply_nested_with_float
deeply_nested_with_float return_deeply_nested_float(deeply_nested_with_float x) {
    return x;
}

// --- Edge cases ---

// Single char
typedef struct single_char {
    char c;
} single_char;

// CHECK-LABEL: define{{.*}} i8 @return_single_char(i8 %x.coerce)
single_char return_single_char(single_char x) {
    return x;
}

// Three chars (24 bits, rounds up to 32)
typedef struct three_chars {
    char a, b, c;
} three_chars;

// CHECK-LABEL: define{{.*}} i32 @return_three_chars(i32 %x.coerce)
three_chars return_three_chars(three_chars x) {
    return x;
}

// Five chars (40 bits, rounds up to 64)
typedef struct five_chars {
    char a, b, c, d, e;
} five_chars;

// CHECK-LABEL: define{{.*}} [2 x i32] @return_five_chars([2 x i32] %x.coerce)
five_chars return_five_chars(five_chars x) {
    return x;
}

// --- Union tests ---

typedef union char_int_union {
    char c;
    int i;
} char_int_union;

// Union with int - preserves union type
// CHECK-LABEL: define{{.*}} %union.char_int_union @return_char_int_union(i32 %x.coerce)
char_int_union return_char_int_union(char_int_union x) {
    return x;
}

typedef union float_int_union {
    float f;
    int i;
} float_int_union;

// Union with float - preserves union type
// CHECK-LABEL: define{{.*}} %union.float_int_union @return_float_int_union(float %x.coerce)
float_int_union return_float_int_union(float_int_union x) {
    return x;
}

// --- Padding scenarios ---

typedef struct char_with_padding {
    char c;
    // 3 bytes padding
    int i;
} char_with_padding;

// Has int, should NOT be coerced even though small + padding
// CHECK-LABEL: define{{.*}} %struct.char_with_padding @return_char_with_padding(i8 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.char_with_padding
char_with_padding return_char_with_padding(char_with_padding x) {
    return x;
}

// ============================================================================
// SECTION 6: Additional exotic aggregates
// ============================================================================

// --- Bitfields ---

typedef struct bitfield_small {
    unsigned a : 4;
    unsigned b : 4;
    unsigned c : 8;
} bitfield_small;

// Bitfields with small bit-widths should be coerced to i32
// Even though backing type is 'unsigned' (32 bits), the actual bit-widths are 4+4+8=16 bits
// CHECK-LABEL: define{{.*}} i32 @return_bitfield_small(i32 %x.coerce)
bitfield_small return_bitfield_small(bitfield_small x) {
    return x;
}

typedef struct bitfield_chars {
    char a : 4;
    char b : 4;
} bitfield_chars;

// Bitfields with char backing type (8-bit) - should be coerced to i16
// CHECK-LABEL: define{{.*}} i16 @return_bitfield_chars(i16 %x.coerce)
bitfield_chars return_bitfield_chars(bitfield_chars x) {
    return x;
}

typedef struct bitfield_with_int {
    unsigned a : 4;
    unsigned b : 4;
    int i;
} bitfield_with_int;

// Bitfields + full int - should NOT be coerced
// Bitfield packs into i8, then padding, then i32
// CHECK-LABEL: define{{.*}} %struct.bitfield_with_int @return_bitfield_with_int(i8 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.bitfield_with_int
bitfield_with_int return_bitfield_with_int(bitfield_with_int x) {
    return x;
}

typedef struct bitfield_with_float {
    unsigned a : 16;
    float f;
} bitfield_with_float;

// Bitfield + float - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.bitfield_with_float @return_bitfield_with_float(i16 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.bitfield_with_float
bitfield_with_float return_bitfield_with_float(bitfield_with_float x) {
    return x;
}

// Bitfields that fill wider ints (up to i64) should also be packed
typedef struct bitfield_large {
    unsigned long long a : 40;
    unsigned long long b : 20;
} bitfield_large;

// 40 + 20 = 60 bits, fits in 64-bit storage - should be coerced to [2 x i32]
// CHECK-LABEL: define{{.*}} [2 x i32] @return_bitfield_large([2 x i32] %x.coerce)
bitfield_large return_bitfield_large(bitfield_large x) {
    return x;
}

typedef struct bitfield_exactly_32 {
    unsigned a : 16;
    unsigned b : 16;
} bitfield_exactly_32;

// 16 + 16 = 32 bits exactly - should be coerced to i32
// CHECK-LABEL: define{{.*}} i32 @return_bitfield_exactly_32(i32 %x.coerce)
bitfield_exactly_32 return_bitfield_exactly_32(bitfield_exactly_32 x) {
    return x;
}

typedef struct bitfield_48 {
    unsigned long long a : 32;
    unsigned long long b : 16;
} bitfield_48;

// 32 + 16 = 48 bits, stored in 64-bit - should be coerced to [2 x i32]
// CHECK-LABEL: define{{.*}} [2 x i32] @return_bitfield_48([2 x i32] %x.coerce)
bitfield_48 return_bitfield_48(bitfield_48 x) {
    return x;
}

// --- _Bool fields ---

typedef struct bool_struct {
    _Bool a;
    _Bool b;
    _Bool c;
    _Bool d;
} bool_struct;

// 4 bools = 32 bits, all sub-32-bit - should be coerced to i32
// CHECK-LABEL: define{{.*}} i32 @return_bool_struct(i32 %x.coerce)
bool_struct return_bool_struct(bool_struct x) {
    return x;
}

typedef struct bool_and_float {
    _Bool b;
    float f;
} bool_and_float;

// Bool + float - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.bool_and_float @return_bool_and_float(i8 %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.bool_and_float
bool_and_float return_bool_and_float(bool_and_float x) {
    return x;
}

typedef struct bool_and_int {
    _Bool b;
    int i;
} bool_and_int;

// Bool + int - should NOT be coerced (int is full-sized)
// CHECK-LABEL: define{{.*}} %struct.bool_and_int @return_bool_and_int(i8 %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.bool_and_int
bool_and_int return_bool_and_int(bool_and_int x) {
    return x;
}

// --- Half-precision floats ---

typedef struct half_struct {
    __fp16 a;
    __fp16 b;
} half_struct;

// Two halfs = 32 bits, but floats - should NOT be coerced
// Two halfs = 32 bits - can be packed into <2 x half> vector
// CHECK-LABEL: define{{.*}} <2 x half> @return_half_struct(<2 x half> %x.coerce)
half_struct return_half_struct(half_struct x) {
    return x;
}

typedef struct half_and_char {
    __fp16 h;
    char c;
} half_and_char;

// Half + char - should NOT be coerced (half is float type)
// CHECK-LABEL: define{{.*}} %struct.half_and_char @return_half_and_char(half %x.coerce0, i8 %x.coerce1)
// CHECK: ret %struct.half_and_char
half_and_char return_half_and_char(half_and_char x) {
    return x;
}

typedef struct four_halfs {
    __fp16 a, b, c, d;
} four_halfs;

// Four halfs = 64 bits - should NOT be coerced
// Four halfs = 64 bits - can be packed into <4 x half> vector
// CHECK-LABEL: define{{.*}} <4 x half> @return_four_halfs(<4 x half> %x.coerce)
four_halfs return_four_halfs(four_halfs x) {
    return x;
}

// --- Bfloat16 tests ---

typedef struct bfloat_struct {
    __bf16 a;
    __bf16 b;
} bfloat_struct;

// Two bfloats = 32 bits, but floats - should NOT be coerced
// Two bfloats = 32 bits - can be packed into <2 x bfloat> vector
// CHECK-LABEL: define{{.*}} <2 x bfloat> @return_bfloat_struct(<2 x bfloat> %x.coerce)
bfloat_struct return_bfloat_struct(bfloat_struct x) {
    return x;
}

typedef struct bfloat_and_char {
    __bf16 b;
    char c;
} bfloat_and_char;

// Bfloat + char - should NOT be coerced (bfloat is float type)
// CHECK-LABEL: define{{.*}} %struct.bfloat_and_char @return_bfloat_and_char(bfloat %x.coerce0, i8 %x.coerce1)
// CHECK: ret %struct.bfloat_and_char
bfloat_and_char return_bfloat_and_char(bfloat_and_char x) {
    return x;
}

typedef struct four_bfloats {
    __bf16 a, b, c, d;
} four_bfloats;

// Four bfloats = 64 bits - should NOT be coerced
// Four bfloats = 64 bits - can be packed into <4 x bfloat> vector
// CHECK-LABEL: define{{.*}} <4 x bfloat> @return_four_bfloats(<4 x bfloat> %x.coerce)
four_bfloats return_four_bfloats(four_bfloats x) {
    return x;
}

// --- Mixed half and bfloat ---

typedef struct mixed_half_bfloat {
    __fp16 h;
    __bf16 b;
} mixed_half_bfloat;

// Mixed half + bfloat - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.mixed_half_bfloat @return_mixed_half_bfloat(half %x.coerce0, bfloat %x.coerce1)
// CHECK: ret %struct.mixed_half_bfloat
mixed_half_bfloat return_mixed_half_bfloat(mixed_half_bfloat x) {
    return x;
}

typedef struct bfloat_and_float {
    __bf16 b;
    float f;
} bfloat_and_float;

// Bfloat + float - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.bfloat_and_float @return_bfloat_and_float(bfloat %x.coerce0, float %x.coerce1)
// CHECK: ret %struct.bfloat_and_float
bfloat_and_float return_bfloat_and_float(bfloat_and_float x) {
    return x;
}

// --- Vectors inside structs ---

typedef int int2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef char char4 __attribute__((ext_vector_type(4)));

typedef struct vec_int2_struct {
    int2 v;
} vec_int2_struct;

// Single-element vector struct - unwrapped to vector type
// CHECK-LABEL: define{{.*}} <2 x i32> @return_vec_int2(<2 x i32> %x.coerce)
vec_int2_struct return_vec_int2(vec_int2_struct x) {
    return x;
}

typedef struct vec_float2_struct {
    float2 v;
} vec_float2_struct;

// Single-element vector struct - unwrapped to vector type
// CHECK-LABEL: define{{.*}} <2 x float> @return_vec_float2(<2 x float> %x.coerce)
vec_float2_struct return_vec_float2(vec_float2_struct x) {
    return x;
}

typedef struct vec_char4_struct {
    char4 v;
} vec_char4_struct;

// Single-element vector struct - unwrapped to vector type
// CHECK-LABEL: define{{.*}} <4 x i8> @return_vec_char4(<4 x i8> %x.coerce)
vec_char4_struct return_vec_char4(vec_char4_struct x) {
    return x;
}

typedef struct vec_and_scalar {
    char4 v;
    int i;
} vec_and_scalar;

// Vector + scalar - should NOT be coerced (vector is not a packable integer type)
// CHECK-LABEL: define{{.*}} %struct.vec_and_scalar @return_vec_and_scalar(<4 x i8> %x.coerce0, i32 %x.coerce1)
// CHECK: ret %struct.vec_and_scalar
vec_and_scalar return_vec_and_scalar(vec_and_scalar x) {
    return x;
}

// --- Arrays of nested structs ---

typedef struct inner_two_chars {
    char a, b;
} inner_two_chars;

typedef struct array_of_nested_chars {
    inner_two_chars arr[2];
} array_of_nested_chars;

// Array of 2 nested structs, each with 2 chars = 32 bits total - should be coerced
// CHECK-LABEL: define{{.*}} i32 @return_array_of_nested_chars(i32 %x.coerce)
array_of_nested_chars return_array_of_nested_chars(array_of_nested_chars x) {
    return x;
}

typedef struct inner_char_float {
    char c;
    float f;
} inner_char_float;

typedef struct array_of_nested_floats {
    inner_char_float arr[1];
} array_of_nested_floats;

// Array of nested struct containing float - should NOT be coerced
// CHECK-LABEL: define{{.*}} %struct.array_of_nested_floats @return_array_of_nested_floats([1 x %struct.inner_char_float] %x.coerce)
// CHECK: ret %struct.array_of_nested_floats
array_of_nested_floats return_array_of_nested_floats(array_of_nested_floats x) {
    return x;
}

typedef struct nested_array_of_shorts {
    struct {
        short arr[2];
    } inner;
} nested_array_of_shorts;

// Nested struct with array of shorts = 32 bits - should be coerced
// CHECK-LABEL: define{{.*}} i32 @return_nested_array_of_shorts(i32 %x.coerce)
nested_array_of_shorts return_nested_array_of_shorts(nested_array_of_shorts x) {
    return x;
}
