// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64,NOHFAALIGN
// RUN: %clang_cc1 -triple arm64-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG64,NOHFAALIGN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32,NOHFAALIGN
// RUN: %clang_cc1 -triple arm64_32-apple-ios7.0 -target-abi darwinpcs -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DARWIN,LONG32,NOHFAALIGN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG64,AAPCS64 --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN --implicit-check-not="not yet implemented"
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -fenable-matrix -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefixes=CHECK,AAPCS,LONG32,NOHFAALIGN --implicit-check-not="not yet implemented"

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
