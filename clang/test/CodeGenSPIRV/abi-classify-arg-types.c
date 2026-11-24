// Checks how non-kernel SPIR-V functions pass arguments.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fexperimental-abi-lowering -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s %}

typedef struct {} empty_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { char a, b, c; } small_t;
typedef struct { int data[17]; } big_t;
typedef union { int i; float f; } int_float_t;
typedef union {
  int *ip;
  float *fp;
} __attribute__((transparent_union)) transparent_t;
typedef short short4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));

// Sub-word integers and bool carry signext or zeroext per their signedness.
// CHECK: define spir_func void @arg_bool(i1 noundef zeroext %{{[^,)]+}})
void arg_bool(_Bool b) {}

// CHECK: define spir_func void @arg_char(i8 noundef signext %{{[^,)]+}})
void arg_char(char c) {}

// CHECK: define spir_func void @arg_uchar(i8 noundef zeroext %{{[^,)]+}})
void arg_uchar(unsigned char c) {}

// CHECK: define spir_func void @arg_short(i16 noundef signext %{{[^,)]+}})
void arg_short(short s) {}

// CHECK: define spir_func void @arg_ushort(i16 noundef zeroext %{{[^,)]+}})
void arg_ushort(unsigned short s) {}

// The call site carries the same extension as the callee.
// CHECK-LABEL: define spir_func void @call_char(i8 noundef signext %{{[^,)]+}})
// CHECK:   call spir_func void @arg_char(i8 noundef signext %{{[^,)]+}})
void call_char(char c) { arg_char(c); }

// Word-sized and wider scalars, pointers and vectors pass through unchanged.
// CHECK: define spir_func void @arg_int(i32 noundef %{{[^,)]+}})
void arg_int(int i) {}

// CHECK: define spir_func void @arg_long(i64 noundef %{{[^,)]+}})
void arg_long(long l) {}

// CHECK: define spir_func void @arg_half(half noundef %{{[^,)]+}})
void arg_half(_Float16 h) {}

// CHECK: define spir_func void @arg_float(float noundef %{{[^,)]+}})
void arg_float(float f) {}

// CHECK: define spir_func void @arg_double(double noundef %{{[^,)]+}})
void arg_double(double d) {}

// Pointers keep their address space outside kernels.
// CHECK: define spir_func void @arg_ptr(ptr noundef %{{[^,)]+}})
void arg_ptr(int *p) {}

// CHECK: define spir_func void @arg_short4(<4 x i16> noundef %{{[^,)]+}})
void arg_short4(short4 v) {}

// CHECK: define spir_func void @arg_float3(<3 x float> noundef %{{[^,)]+}})
void arg_float3(float3 v) {}

// SPIR-V has no 128-bit integer, so a _BitInt wider than 64 bits is passed
// indirectly.
// CHECK: define spir_func void @arg_bitint64(i64 noundef %{{[^,)]+}})
void arg_bitint64(_BitInt(64) i) {}

// CHECK: define spir_func void @arg_bitint65(ptr noundef byval(i128) align 8 %{{[^,)]+}})
void arg_bitint65(_BitInt(65) i) {}

// Every aggregate is copied byval, including an empty one.
// CHECK: define spir_func void @arg_empty(ptr noundef byval(%struct.empty_t) align 1 %{{[^,)]+}}, i32 noundef %{{[^,)]+}})
void arg_empty(empty_t e, int i) {}

// CHECK: define spir_func void @arg_i32_wrapper(ptr noundef byval(%struct.i32_wrapper_t) align 4 %{{[^,)]+}})
void arg_i32_wrapper(i32_wrapper_t w) {}

// CHECK: define spir_func void @arg_small(ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}})
void arg_small(small_t s) {}

// CHECK: define spir_func void @arg_big(ptr noundef byval(%struct.big_t) align 4 %{{[^,)]+}})
void arg_big(big_t b) {}

// CHECK: define spir_func void @arg_union(ptr noundef byval(%union.int_float_t) align 4 %{{[^,)]+}})
void arg_union(int_float_t u) {}

// CHECK: define spir_func void @arg_cfloat(ptr noundef byval({ float, float }) align 4 %{{[^,)]+}})
void arg_cfloat(_Complex float c) {}

// CHECK: define spir_func void @arg_cdouble(ptr noundef byval({ double, double }) align 8 %{{[^,)]+}})
void arg_cdouble(_Complex double c) {}

// A transparent union is passed as its first member.
// CHECK: define spir_func void @arg_transparent(ptr %{{[^,)]+}})
void arg_transparent(transparent_t u) {}

// Call sites, direct and indirect, pass the same byval pointer.
// CHECK-LABEL: define spir_func void @call_small(ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}})
// CHECK:   call spir_func void @arg_small(ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}})
void call_small(small_t s) { arg_small(s); }

// CHECK-LABEL: define spir_func void @call_fnptr(ptr noundef %{{[^,)]+}}, ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}})
// CHECK:   call spir_func void %{{.*}}(ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}})
void call_fnptr(void (*fp)(small_t), small_t s) { fp(s); }

void variadic(int n, ...);

// Variadic arguments are classified like fixed ones, after default argument
// promotion.
// CHECK-LABEL: define spir_func void @call_variadic(ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}}, i8 noundef signext %{{[^,)]+}})
// CHECK:   call spir_func void (i32, ...) @variadic(i32 noundef 2, ptr noundef byval(%struct.small_t) align 1 %{{[^,)]+}}, i32 noundef %{{[^,)]+}})
void call_variadic(small_t s, char c) { variadic(2, s, c); }
