// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64,NOHFAALIGN,NOHUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64,NOHFAALIGN,NOHUGEVEC,NOANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32,NOHFAALIGN,HUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32,NOHFAALIGN,HUGEVEC,NOANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,NOANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,NOANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-android -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,ANDROID
// RUN: %clang_cc1 -triple aarch64-linux-android -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64,NOHUGEVEC,ANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN,NOHUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN,NOHUGEVEC,NOANDROID --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN,NOHUGEVEC,NOANDROID
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN,NOHUGEVEC,NOANDROID --implicit-check-not="not yet implemented"

// This test is verifying that the LLVM ABI library classifies argument types in
// the same way that Clang does without the library.

// The AArch64 support in the ABI library is a work in progress. New test cases
// will be added here as the types are implemented. Unimplemented cases will
// report a warning if the ABI library is used.

void arg_void(void) {
}
// CHECK: define{{.*}} void @arg_void()

void arg_bool(_Bool b) {}
// AAPCS: define{{.*}} void @arg_bool(i1 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_bool(i1 noundef zeroext %{{.*}})

void arg_char(char c) {}
// AAPCS: define{{.*}} void @arg_char(i8 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_char(i8 noundef signext %{{.*}})

void arg_short(short s) {}
// AAPCS: define{{.*}} void @arg_short(i16 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_short(i16 noundef signext %{{.*}})

void arg_ushort(unsigned short us) {}
// AAPCS: define{{.*}} void @arg_ushort(i16 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_ushort(i16 noundef zeroext %{{.*}})

void arg_int(int i) {}
// CHECK: define{{.*}} void @arg_int(i32 noundef %{{.*}})

void arg_uint(unsigned int ui) {}
// CHECK: define{{.*}} void @arg_uint(i32 noundef %{{.*}})

void arg_long(long int li) {}
// LONG64: define{{.*}} void @arg_long(i64 noundef %{{.*}})
// LONG32: define{{.*}} void @arg_long(i32 noundef %{{.*}})

void arg_float16(_Float16 f16) {}
// CHECK: define{{.*}} void @arg_float16(half noundef %{{.*}})

void arg_fp16(__fp16 f16) {}
// CHECK: define{{.*}} void @arg_fp16(half noundef %{{.*}})

void arg_float(float f) {}
// CHECK: define{{.*}} void @arg_float(float noundef %{{.*}})

void arg_double(double d) {}
// CHECK: define{{.*}} void @arg_double(double noundef %{{.*}})

int gi;
void arg_int_ptr(int* pi) {}
// CHECK: define{{.*}} void @arg_int_ptr(ptr noundef %{{.*}})

void arg_void_ptr(void* pv) {}
// CHECK: define{{.*}} void @arg_void_ptr(ptr noundef %{{.*}})

typedef float fx2x2_t __attribute__((matrix_type(2, 2)));
void arg_matrix(fx2x2_t m) {}
// CHECK: define{{.*}} void @arg_matrix(<4 x float> noundef %{{.*}})

// Transparent unions are passed as their first field.
typedef union {
  int i;
  float f;
} tu_int_t __attribute__((transparent_union));
void arg_transparent_union_int(tu_int_t tu) {}
// CHECK: define{{.*}} void @arg_transparent_union_int(i32 %{{.*}})

typedef union {
  char c;
  signed char sc;
} tu_char_t __attribute__((transparent_union));
void arg_transparent_union_char(tu_char_t tu) {}
// AAPCS: define{{.*}} void @arg_transparent_union_char(i8 %{{.*}})
// DARWIN: define{{.*}} void @arg_transparent_union_char(i8 noundef signext %{{.*}})

typedef union {
  void *p;
  int *ip;
} tu_ptr_t __attribute__((transparent_union));
void arg_transparent_union_ptr(tu_ptr_t tu) {}
// CHECK: define{{.*}} void @arg_transparent_union_ptr(ptr %{{.*}})

void arg_bitint7(_BitInt(7) x) {}
// AAPCS: define{{.*}} void @arg_bitint7(i7 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_bitint7(i7 noundef signext %{{.*}})

void arg_ubitint7(unsigned _BitInt(7) x) {}
// AAPCS: define{{.*}} void @arg_ubitint7(i7 noundef %{{.*}})
// DARWIN: define{{.*}} void @arg_ubitint7(i7 noundef zeroext %{{.*}})

void arg_bitint65(_BitInt(65) x) {}
// CHECK: define{{.*}} void @arg_bitint65(i65 noundef %{{.*}})

void arg_bitint128(_BitInt(128) x) {}
// CHECK: define{{.*}} void @arg_bitint128(i128 noundef %{{.*}})

void arg_bitint129(_BitInt(129) x) {}
// CHECK: define{{.*}} void @arg_bitint129(ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{.*}})

// Homogeneous floating-point aggregates are coerced to an array of the base
// type. AAPCS sets alignstack from unadjusted alignment (8, or 16 if the
// unadjusted alignment is at least 16). DarwinPCS and Win64 do not.

typedef struct {
  float a, b;
} HFA2f;
void arg_hfa2f(HFA2f h) {}
// AAPCS64: define{{.*}} void @arg_hfa2f([2 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa2f([2 x float] %{{.*}})

typedef struct {
  double a, b, c, d;
} HFA4d;
void arg_hfa4d(HFA4d h) {}
// AAPCS64: define{{.*}} void @arg_hfa4d([4 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa4d([4 x double] %{{.*}})

typedef struct {
  float v[3];
} HFA3arr;
void arg_hfa3arr(HFA3arr h) {}
// AAPCS64: define{{.*}} void @arg_hfa3arr([3 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa3arr([3 x float] %{{.*}})

typedef struct {
  _Float16 a, b;
} HFA2h;
void arg_hfa2h(HFA2h h) {}
// AAPCS64: define{{.*}} void @arg_hfa2h([2 x half] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa2h([2 x half] %{{.*}})

typedef struct {
  HFA2f inner;
  float c;
} HFANested;
void arg_hfa_nested(HFANested h) {}
// AAPCS64: define{{.*}} void @arg_hfa_nested([3 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_nested([3 x float] %{{.*}})

typedef struct {
  int : 0;
  float a, b;
} HFAZeroBF;
void arg_hfa_zerobf(HFAZeroBF h) {}
// AAPCS64: define{{.*}} void @arg_hfa_zerobf([2 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_zerobf([2 x float] %{{.*}})

typedef union {
  float a;
  float v[3];
} HFAUnion;
void arg_hfa_union(HFAUnion h) {}
// AAPCS64: define{{.*}} void @arg_hfa_union([3 x float] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hfa_union([3 x float] %{{.*}})

void arg_complex_float(_Complex float c) {}
// AAPCS64: define{{.*}} void @arg_complex_float([2 x float] {{(noundef )?}}alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_complex_float([2 x float] {{(noundef )?}}%{{.*}})

typedef float f32x2 __attribute__((vector_size(8)));
typedef float f32x4 __attribute__((vector_size(16)));

typedef struct {
  f32x2 a, b;
} HVA2x64;
void arg_hva2x64(HVA2x64 h) {}
// AAPCS64: define{{.*}} void @arg_hva2x64([2 x <2 x float>] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hva2x64([2 x <2 x float>] %{{.*}})

typedef struct {
  f32x4 a, b;
} HVA2x128;
void arg_hva2x128(HVA2x128 h) {}
// AAPCS64: define{{.*}} void @arg_hva2x128([2 x <4 x float>] alignstack(16) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_hva2x128([2 x <4 x float>] %{{.*}})

// Record-level aligned(16) on a 16-byte HFA raises ABI alignment to 16 without
// adding padding, so the type is still homogeneous. Unadjusted alignment is
// still 8, so AAPCS must use alignstack(8), not 16.
typedef struct __attribute__((aligned(16))) {
  double a, b;
} OveralignedHFA;
void arg_overaligned_hfa(OveralignedHFA h) {}
// AAPCS64: define{{.*}} void @arg_overaligned_hfa([2 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_hfa([2 x double] %{{.*}})

// aligned(32) on a 32-byte HFA likewise stays homogeneous. Unadjusted
// alignment is 8, so AAPCS must not take the 16-byte cap.
typedef struct __attribute__((aligned(32))) {
  double a, b, c, d;
} Overaligned32HFA;
void arg_overaligned32_hfa(Overaligned32HFA h) {}
// AAPCS64: define{{.*}} void @arg_overaligned32_hfa([4 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned32_hfa([4 x double] %{{.*}})

// The unadjusted alignment of a union is tracked the same way. The widest
// member is already 16 bytes, so aligned(16) adds no padding.
typedef union __attribute__((aligned(16))) {
  double a;
  double v[2];
} OveralignedUnionHFA;
void arg_overaligned_union_hfa(OveralignedUnionHFA u) {}
// AAPCS64: define{{.*}} void @arg_overaligned_union_hfa([2 x double] alignstack(8) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_overaligned_union_hfa([2 x double] %{{.*}})

// Field alignment is part of unadjusted alignment, so AAPCS uses 16.
typedef struct {
  __attribute__((aligned(16))) double v[2];
} FieldAlignedHFA;
void arg_field_aligned_hfa(FieldAlignedHFA h) {}
// AAPCS64: define{{.*}} void @arg_field_aligned_hfa([2 x double] alignstack(16) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_field_aligned_hfa([2 x double] %{{.*}})

// Unadjusted alignment of 32 is capped at 16.
typedef struct {
  __attribute__((aligned(32))) double v[4];
} FieldAligned32HFA;
void arg_field_aligned32_hfa(FieldAligned32HFA h) {}
// AAPCS64: define{{.*}} void @arg_field_aligned32_hfa([4 x double] alignstack(16) %{{.*}})
// NOHFAALIGN: define{{.*}} void @arg_field_aligned32_hfa([4 x double] %{{.*}})

// Empty records and zero-size types are ignored as arguments in C.
typedef struct {
} Empty;
void arg_empty(Empty e) {}
// CHECK: define{{.*}} void @arg_empty()

void arg_empty_then_int(Empty e, int i) {}
// CHECK: define{{.*}} void @arg_empty_then_int(i32 noundef %{{.*}})

typedef union {
} EmptyUnion;
void arg_empty_union(EmptyUnion u) {}
// CHECK: define{{.*}} void @arg_empty_union()

typedef struct {
  int arr[0];
} ZeroSize;
void arg_zerosize(ZeroSize z) {}
// CHECK: define{{.*}} void @arg_zerosize()

typedef struct {
  ZeroSize inner;
} NestedZeroSize;
void arg_nested_zerosize(NestedZeroSize z) {}
// CHECK: define{{.*}} void @arg_nested_zerosize()

// Legal 64- and 128-bit vectors are passed directly. Illegal vectors are
// coerced to an integer or integer vector, or passed indirectly if larger
// than 128 bits. arm64_32 MachO treats vectors larger than 32 bits as legal.

typedef float v2f32 __attribute__((vector_size(8)));
void arg_v2f32(v2f32 v) {}
// CHECK: define{{.*}} void @arg_v2f32(<2 x float> noundef %{{.*}})

typedef float v4f32 __attribute__((vector_size(16)));
void arg_v4f32(v4f32 v) {}
// CHECK: define{{.*}} void @arg_v4f32(<4 x float> noundef %{{.*}})

typedef char v16i8 __attribute__((vector_size(16)));
void arg_v16i8(v16i8 v) {}
// CHECK: define{{.*}} void @arg_v16i8(<16 x i8> noundef %{{.*}})

typedef char v2i8 __attribute__((vector_size(2)));
void arg_v2i8(v2i8 v) {}
// ANDROID: define{{.*}} void @arg_v2i8(i16 noundef %{{.*}})
// NOANDROID: define{{.*}} void @arg_v2i8(i32{{.*}} %{{.*}})

typedef char v3i8 __attribute__((vector_size(3)));
void arg_v3i8(v3i8 v) {}
// CHECK: define{{.*}} void @arg_v3i8(i32{{.*}} %{{.*}})

typedef char v4i8 __attribute__((vector_size(4)));
void arg_v4i8(v4i8 v) {}
// CHECK: define{{.*}} void @arg_v4i8(i32{{.*}} %{{.*}})

typedef unsigned __int128 v1i128 __attribute__((vector_size(16)));
void arg_v1i128(v1i128 v) {}
// NOHUGEVEC: define{{.*}} void @arg_v1i128(<4 x i32> noundef %{{.*}})
// HUGEVEC: define{{.*}} void @arg_v1i128(<1 x i128> noundef %{{.*}})

typedef float v8f32 __attribute__((vector_size(32)));
void arg_v8f32(v8f32 v) {}
// NOHUGEVEC: define{{.*}} void @arg_v8f32(ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{.*}})
// HUGEVEC: define{{.*}} void @arg_v8f32(<8 x float> noundef %{{.*}})

typedef char v17i8 __attribute__((vector_size(17)));
void arg_v17i8(v17i8 v) {}
// CHECK: define{{.*}} void @arg_v17i8(ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{.*}})

// A vector whose element count is not a power of 2 is illegal, and it is
// coerced based on its ABI size, which is the payload width rounded up to a
// power of 2. So a 3 x float has 96 bits of payload but is coerced as if it
// were 128 bits wide.

typedef float v3f32 __attribute__((vector_size(12)));
void arg_v3f32(v3f32 v) {}
// CHECK: define{{.*}} void @arg_v3f32(<4 x i32> %{{.*}})

typedef short v3i16 __attribute__((vector_size(6)));
void arg_v3i16(v3i16 v) {}
// CHECK: define{{.*}} void @arg_v3i16(<2 x i32> %{{.*}})

typedef char v5i8 __attribute__((vector_size(5)));
void arg_v5i8(v5i8 v) {}
// CHECK: define{{.*}} void @arg_v5i8(<2 x i32> %{{.*}})

typedef char v9i8 __attribute__((vector_size(9)));
void arg_v9i8(v9i8 v) {}
// CHECK: define{{.*}} void @arg_v9i8(<4 x i32> %{{.*}})

// A _BitInt occupies a whole number of bytes, so a sub-byte element counts as
// 8 bits towards the size of the vector. That makes 8 x _BitInt(2) a legal
// 64-bit vector rather than an illegal 16-bit one.

typedef _BitInt(2) b2v4 __attribute__((ext_vector_type(4)));
void arg_b2v4(b2v4 v) {}
// CHECK: define{{.*}} void @arg_b2v4(i32 %{{.*}})

typedef _BitInt(2) b2v8 __attribute__((ext_vector_type(8)));
void arg_b2v8(b2v8 v) {}
// CHECK: define{{.*}} void @arg_b2v8(<8 x i2> noundef %{{.*}})

typedef _BitInt(4) b4v16 __attribute__((ext_vector_type(16)));
void arg_b4v16(b4v16 v) {}
// CHECK: define{{.*}} void @arg_b4v16(<16 x i4> noundef %{{.*}})

typedef _BitInt(32) b32v2 __attribute__((ext_vector_type(2)));
void arg_b32v2(b32v2 v) {}
// CHECK: define{{.*}} void @arg_b32v2(<2 x i32> noundef %{{.*}})
