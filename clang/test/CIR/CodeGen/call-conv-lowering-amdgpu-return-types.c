// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies return values of ordinary AMDGPU
// functions, which use the default convention rather than the kernel one.

// TODO(cir): Add the return cases still NYI in CallConvLowering:
// - Aggregates of 33 to 64 bits, which classic coerces to [2 x i32].
//   abiTypeToCIR has no array case yet.
// - Aggregates over 64 bits that fit the 16-register budget. The classifier
//   returns Direct with no coerce type, which the bridge rejects.
// - Larger aggregates, returned through sret in addrspace(5). The bridge
//   drops the indirect address space.
// - Structs with a flexible array member. The bridge does not set that
//   record flag yet.
// - 3-element vector returns, blocked on CIRGen's vec3 load and store NYI.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { int *p; } ptr_wrapper_t;
typedef struct { char a, b; } char2_t;
typedef struct { char a, b, c; } small_t;
typedef union { int i; float f; } int_float_t;

void ret_void(void) {}

// CIR: cir.func {{.*}}@ret_void()
// LLVM: define {{.*}}void @ret_void()

// Sub-word integers and bool carry signext or zeroext per their signedness.
_Bool ret_bool(void) { return 0; }

// CIR: cir.func {{.*}}@ret_bool() -> (!cir.bool {{.*}}llvm.zeroext
// LLVM: define {{.*}}zeroext i1 @ret_bool()

char ret_char(void) { return 0; }

// CIR: cir.func {{.*}}@ret_char() -> (!s8i {{.*}}llvm.signext
// LLVM: define {{.*}}signext i8 @ret_char()

unsigned char ret_uchar(void) { return 0; }

// CIR: cir.func {{.*}}@ret_uchar() -> (!u8i {{.*}}llvm.zeroext
// LLVM: define {{.*}}zeroext i8 @ret_uchar()

short ret_short(void) { return 0; }

// CIR: cir.func {{.*}}@ret_short() -> (!s16i {{.*}}llvm.signext
// LLVM: define {{.*}}signext i16 @ret_short()

// Word-sized and wider scalars and pointers return unchanged.
int ret_int(void) { return 0; }

// CIR: cir.func {{.*}}@ret_int() -> {{\(?}}!s32i
// LLVM: define {{.*}}i32 @ret_int()

long ret_long(void) { return 0; }

// CIR: cir.func {{.*}}@ret_long() -> {{\(?}}!s64i
// LLVM: define {{.*}}i64 @ret_long()

float ret_float(void) { return 0; }

// CIR: cir.func {{.*}}@ret_float() -> {{\(?}}!cir.float
// LLVM: define {{.*}}float @ret_float()

double ret_double(void) { return 0; }

// CIR: cir.func {{.*}}@ret_double() -> {{\(?}}!cir.double
// LLVM: define {{.*}}double @ret_double()

_Float16 ret_half(void) { return 0; }

// CIR: cir.func {{.*}}@ret_half() -> {{\(?}}!cir.f16
// LLVM: define {{.*}}half @ret_half()

int *ret_ptr(void) { return 0; }

// CIR: cir.func {{.*}}@ret_ptr() -> {{\(?}}!cir.ptr<!s32i>
// LLVM: define {{.*}}ptr @ret_ptr()
_BitInt(65) ret_bitint65(void) { return 0; }

// CIR: cir.func {{.*}}@ret_bitint65() -> {{\(?}}!cir.int<s, 65, bitint>
// LLVM: define {{.*}}i65 @ret_bitint65()

// An empty struct return becomes void.
empty_t ret_empty(void) {
  empty_t e;
  return e;
}

// CIR: cir.func {{.*}}@ret_empty()
// LLVM: define {{.*}}void @ret_empty()

// Single-element structs return as their element.
i8_wrapper_t ret_i8_wrapper(void) {
  i8_wrapper_t w = {1};
  return w;
}

// CIR: cir.func {{.*}}@ret_i8_wrapper() -> !s8i
// LLVM: define {{.*}}i8 @ret_i8_wrapper()

i32_wrapper_t ret_i32_wrapper(void) {
  i32_wrapper_t w = {1};
  return w;
}

// CIR: cir.func {{.*}}@ret_i32_wrapper() -> !s32i
// LLVM: define {{.*}}i32 @ret_i32_wrapper()

f32_wrapper_t ret_f32_wrapper(void) {
  f32_wrapper_t w = {1.0f};
  return w;
}

// CIR: cir.func {{.*}}@ret_f32_wrapper() -> !cir.float
// LLVM: define {{.*}}float @ret_f32_wrapper()

ptr_wrapper_t ret_ptr_wrapper(void) {
  ptr_wrapper_t w = {0};
  return w;
}

// CIR: cir.func {{.*}}@ret_ptr_wrapper() -> !cir.ptr<{{[^>]*}}>
// LLVM: define {{.*}}ptr @ret_ptr_wrapper()

// Aggregates up to 32 bits are packed into one integer register.
char2_t ret_char2(void) {
  char2_t s = {1, 2};
  return s;
}

// CIR: cir.func {{.*}}@ret_char2() -> !u16i
// LLVM: define {{.*}}i16 @ret_char2()

small_t ret_small(void) {
  small_t s = {1, 2, 3};
  return s;
}

// CIR: cir.func {{.*}}@ret_small() -> !u32i
// LLVM: define {{.*}}i32 @ret_small()

int_float_t ret_union(void) {
  int_float_t u = {1};
  return u;
}

// CIR: cir.func {{.*}}@ret_union() -> !u32i
// LLVM: define {{.*}}i32 @ret_union()
