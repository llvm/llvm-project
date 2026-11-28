// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-llvm %s -o - | FileCheck %s %}

// Checks how non-kernel AMDGPU functions return values.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { int *p; } ptr_wrapper_t;
typedef struct { char a, b; } char2_t;
typedef struct { char a, b, c; } small_t;
typedef union { int i; float f; } int_float_t;
typedef struct { int a, b; } pair_t;
typedef struct { int a, b, c; } triple_t;
typedef struct { int data[17]; } big_t;
typedef struct { int data[16]; } big16_t;
typedef struct { int n; int data[]; } fam_t;
typedef struct { _Complex double c; } cdwrap_t;
typedef union { int a[3]; double d; } u_pad_t;

void ret_void(void) {}

// CHECK: define {{.*}}void @ret_void()

// Sub-word integers and bool carry signext or zeroext per their signedness.
_Bool ret_bool(void) { return 0; }

// CHECK: define {{.*}}zeroext i1 @ret_bool()

char ret_char(void) { return 0; }

// CHECK: define {{.*}}signext i8 @ret_char()

unsigned char ret_uchar(void) { return 0; }

// CHECK: define {{.*}}zeroext i8 @ret_uchar()

short ret_short(void) { return 0; }

// CHECK: define {{.*}}signext i16 @ret_short()

// Word-sized and wider scalars and pointers return unchanged.
int ret_int(void) { return 0; }

// CHECK: define {{.*}}i32 @ret_int()

long ret_long(void) { return 0; }

// CHECK: define {{.*}}i64 @ret_long()

float ret_float(void) { return 0; }

// CHECK: define {{.*}}float @ret_float()

double ret_double(void) { return 0; }

// CHECK: define {{.*}}double @ret_double()

_Float16 ret_half(void) { return 0; }

// CHECK: define {{.*}}half @ret_half()

int *ret_ptr(void) { return 0; }

// CHECK: define {{.*}}ptr @ret_ptr()

_BitInt(65) ret_bitint65(void) { return 0; }

// CHECK: define {{.*}}i65 @ret_bitint65()

// An empty struct return becomes void.
empty_t ret_empty(void) {
  empty_t e;
  return e;
}

// CHECK: define {{.*}}void @ret_empty()

// Single-element structs return as their element.
i8_wrapper_t ret_i8_wrapper(void) {
  i8_wrapper_t w = {1};
  return w;
}

// CHECK: define {{.*}}i8 @ret_i8_wrapper()

i32_wrapper_t ret_i32_wrapper(void) {
  i32_wrapper_t w = {1};
  return w;
}

// CHECK: define {{.*}}i32 @ret_i32_wrapper()

f32_wrapper_t ret_f32_wrapper(void) {
  f32_wrapper_t w = {1.0f};
  return w;
}

// CHECK: define {{.*}}float @ret_f32_wrapper()

ptr_wrapper_t ret_ptr_wrapper(void) {
  ptr_wrapper_t w = {0};
  return w;
}

// CHECK: define {{.*}}ptr @ret_ptr_wrapper()

// A one-field struct return is not unwrapped.
cdwrap_t ret_cdwrap(void) {
  cdwrap_t w = {0};
  return w;
}

// CHECK: define {{.*}}%struct.cdwrap_t @ret_cdwrap()
// CHECK:   %[[RET:.*]] = load %struct.cdwrap_t, ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   ret %struct.cdwrap_t %[[RET]]

// Aggregates up to 32 bits are packed into one integer register.
char2_t ret_char2(void) {
  char2_t s = {1, 2};
  return s;
}

// CHECK: define {{.*}}i16 @ret_char2()

small_t ret_small(void) {
  small_t s = {1, 2, 3};
  return s;
}

// CHECK: define {{.*}}i32 @ret_small()

int_float_t ret_union(void) {
  int_float_t u = {1};
  return u;
}

// CHECK: define {{.*}}i32 @ret_union()

// Aggregates of 33 to 64 bits are packed into a pair of i32 registers.
pair_t ret_pair(void) {
  pair_t p = {1, 2};
  return p;
}

// CHECK: define {{.*}}[2 x i32] @ret_pair()
// CHECK:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// CHECK:   ret [2 x i32] %[[RET]]

// The call site unpacks the result the same way.
void call_ret_pair(void) { pair_t p = ret_pair(); }

// CHECK: define {{.*}}void @call_ret_pair()
// CHECK:   %[[RES:.*]] = call [2 x i32] @ret_pair()
// CHECK:   store [2 x i32] %[[RES]], ptr addrspace(5) %{{.*}}, align 4

// A _Complex up to 64 bits is packed like a small aggregate.
_Complex float ret_cfloat(void) { return 0; }

// CHECK: define {{.*}}[2 x i32] @ret_cfloat()
// CHECK:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// CHECK-NEXT:   ret [2 x i32] %[[RET]]

// Larger aggregates that fit the register budget return whole.
triple_t ret_triple(void) {
  triple_t t = {1, 2, 3};
  return t;
}

// CHECK: define {{.*}}%struct.triple_t @ret_triple()
// CHECK:   %[[RET:.*]] = load %struct.triple_t, ptr addrspace(5) %{{.*}}, align 4
// CHECK:   ret %struct.triple_t %[[RET]]

// A wider _Complex that fits the register budget returns whole.
_Complex double ret_cdouble(void) { return 0; }

// CHECK: define {{.*}}{ double, double } @ret_cdouble()
// CHECK:   ret { double, double } %{{.*}}

// A union return is not flattened.
u_pad_t ret_u_pad(u_pad_t *p) { return *p; }

// CHECK: define {{.*}}%union.u_pad_t @ret_u_pad(ptr noundef %{{.*}})
// CHECK:   %[[RET:.*]] = load %union.u_pad_t, ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   ret %union.u_pad_t %[[RET]]

// An aggregate of exactly 16 registers is returned direct.
big16_t ret_big16(big16_t *p) { return *p; }

// CHECK: define {{.*}}%struct.big16_t @ret_big16(ptr noundef %{{.*}})
// CHECK:   %[[RET:.*]] = load %struct.big16_t, ptr addrspace(5) %{{.*}}, align 4
// CHECK-NEXT:   ret %struct.big16_t %[[RET]]

// An aggregate past the register budget returns through sret in addrspace(5).
big_t ret_big(void) {
  big_t b = {0};
  return b;
}

// CHECK: define {{.*}}void @ret_big(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.big_t) align 4 %[[RET:[^)]+]])
// CHECK:   %{{.*}} = addrspacecast ptr addrspace(5) %[[RET]] to ptr

// The caller's local is passed as the sret.
void call_ret_big(void) { big_t b = ret_big(); }

// CHECK: define {{.*}}void @call_ret_big()
// CHECK:   alloca %struct.big_t, align 4, addrspace(5)
// CHECK:   call void @ret_big(ptr addrspace(5) dead_on_unwind writable sret(%struct.big_t) align 4 %{{.*}})

// A struct with a flexible array member returns through sret in addrspace(5).
fam_t ret_fam(void) {
  fam_t s = {0};
  return s;
}

// CHECK: define {{.*}}void @ret_fam(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.fam_t) align 4 %[[RET:[^)]+]])
// CHECK:   %{{.*}} = addrspacecast ptr addrspace(5) %[[RET]] to ptr
