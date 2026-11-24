// Checks how non-kernel SPIR-V functions return values.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -fexperimental-abi-lowering -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %if cir-enabled %{ %clang_cc1 -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s %}

typedef struct {} empty_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { int data[17]; } big_t;
typedef union { int i; float f; } int_float_t;
typedef short short4 __attribute__((ext_vector_type(4)));

// Sub-word integers and bool carry signext or zeroext per their signedness.
// CHECK: define spir_func zeroext i1 @ret_bool()
_Bool ret_bool(void) { return 1; }

// CHECK: define spir_func signext i8 @ret_char()
char ret_char(void) { return 1; }

// CHECK: define spir_func zeroext i8 @ret_uchar()
unsigned char ret_uchar(void) { return 1; }

// CHECK: define spir_func signext i16 @ret_short()
short ret_short(void) { return 1; }

// CHECK: define spir_func zeroext i16 @ret_ushort()
unsigned short ret_ushort(void) { return 1; }

// The call site carries the same extension as the callee.
// CHECK-LABEL: define spir_func signext i8 @call_ret_char()
// CHECK:   call spir_func signext i8 @ret_char()
char call_ret_char(void) { return ret_char(); }

// Word-sized and wider scalars, pointers and vectors are returned directly.
// CHECK: define spir_func i32 @ret_int()
int ret_int(void) { return 1; }

// CHECK: define spir_func i64 @ret_long()
long ret_long(void) { return 1; }

// CHECK: define spir_func float @ret_float()
float ret_float(void) { return 1; }

// CHECK: define spir_func double @ret_double()
double ret_double(void) { return 1; }

// CHECK: define spir_func ptr @ret_ptr(ptr noundef %{{[^,)]+}})
int *ret_ptr(int *p) { return p; }

// CHECK: define spir_func <4 x i16> @ret_short4(<4 x i16> noundef %{{[^,)]+}})
short4 ret_short4(short4 v) { return v; }

// SPIR-V has no 128-bit integer, so a _BitInt wider than 64 bits is returned
// through sret.
// CHECK: define spir_func i64 @ret_bitint64()
_BitInt(64) ret_bitint64(void) { return 1; }

// CHECK: define spir_func void @ret_bitint65(ptr dead_on_unwind noalias writable sret(i128) align 8 %{{[^,)]+}})
_BitInt(65) ret_bitint65(void) { return 1; }

// CHECK-LABEL: define spir_func i32 @call_ret_bitint65()
// CHECK:   call spir_func void @ret_bitint65(ptr dead_on_unwind writable sret(i128) align 8 %{{[^,)]+}})
int call_ret_bitint65(void) { return (int)ret_bitint65(); }

// Every aggregate is returned through sret in the private address space,
// including an empty one.
// CHECK: define spir_func void @ret_empty(ptr dead_on_unwind noalias writable sret(%struct.empty_t) align 1 %{{[^,)]+}})
empty_t ret_empty(void) { empty_t e; return e; }

// CHECK: define spir_func void @ret_i32_wrapper(ptr dead_on_unwind noalias writable sret(%struct.i32_wrapper_t) align 4 %{{[^,)]+}})
i32_wrapper_t ret_i32_wrapper(void) { i32_wrapper_t w = {1}; return w; }

// CHECK: define spir_func void @ret_big(ptr dead_on_unwind noalias writable sret(%struct.big_t) align 4 %{{[^,)]+}})
big_t ret_big(void) { big_t b = {}; return b; }

// CHECK: define spir_func void @ret_union(ptr dead_on_unwind noalias writable sret(%union.int_float_t) align 4 %{{[^,)]+}})
int_float_t ret_union(void) { int_float_t u = {1}; return u; }

// CHECK: define spir_func void @ret_cfloat(ptr dead_on_unwind noalias writable sret({ float, float }) align 4 %{{[^,)]+}})
_Complex float ret_cfloat(void) { return 1; }

// CHECK-LABEL: define spir_func i32 @call_ret_i32_wrapper()
// CHECK:   call spir_func void @ret_i32_wrapper(ptr dead_on_unwind writable sret(%struct.i32_wrapper_t) align 4 %{{[^,)]+}})
int call_ret_i32_wrapper(void) { return ret_i32_wrapper().i; }
