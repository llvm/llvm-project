// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies return values of ordinary AMDGPU
// functions, which use the default convention rather than the kernel one.

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
typedef struct { int n; int data[]; } fam_t;
typedef struct { _Complex double c; } cdwrap_t;
typedef union { int a[3]; double d; } u_pad_t;

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

// Aggregates of 33 to 64 bits are packed into a pair of i32 registers.
pair_t ret_pair(void) {
  pair_t p = {1, 2};
  return p;
}

// CIR: cir.func {{.*}}@ret_pair() -> !cir.array<!u32i x 2>
// CIR:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_pair_t, target_address_space(5)>
// CIR:   %[[RET_ARR:.*]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_pair_t, target_address_space(5)> -> !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR:   %[[RET:.*]] = cir.load align(4) %[[RET_ARR]] : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>, !cir.array<!u32i x 2>
// CIR:   cir.return %[[RET]] : !cir.array<!u32i x 2>
// LLVM: define {{.*}}[2 x i32] @ret_pair()
// LLVM:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// LLVM:   ret [2 x i32] %[[RET]]

// Larger aggregates that fit the register budget return whole.
triple_t ret_triple(void) {
  triple_t t = {1, 2, 3};
  return t;
}

// CIR: cir.func {{.*}}@ret_triple() -> !rec_triple_t
// CIR:   %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_triple_t, target_address_space(5)>
// CIR:   %[[RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!rec_triple_t, target_address_space(5)>, !rec_triple_t
// CIR:   cir.return %[[RET]] : !rec_triple_t
// LLVM: define {{.*}}%struct.triple_t @ret_triple()
// LLVM:   %[[RET:.*]] = load %struct.triple_t, ptr addrspace(5) %{{.*}}, align 4
// LLVM:   ret %struct.triple_t %[[RET]]

// The call site unpacks the result the same way.
void call_ret_pair(void) { pair_t p = ret_pair(); }

// CIR: cir.func {{.*}}@call_ret_pair()
// CIR:   %[[COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR:   %[[P:.*]] = cir.alloca "p" {{.*}} : !cir.ptr<!rec_pair_t, target_address_space(5)>
// CIR:   %[[P_VIEW:.*]] = cir.cast address_space %[[P]] : !cir.ptr<!rec_pair_t, target_address_space(5)> -> !cir.ptr<!rec_pair_t>
// CIR:   %[[RES:.*]] = cir.call @ret_pair() {{.*}}: () -> !cir.array<!u32i x 2>
// CIR:   cir.store %[[RES]], %[[COERCE]] : !cir.array<!u32i x 2>, !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR:   %[[COERCE_REC:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)> -> !cir.ptr<!rec_pair_t, target_address_space(5)>
// CIR:   cir.copy %[[COERCE_REC]] align(4) to %[[P_VIEW]] align(4)
// LLVM: define {{.*}}void @call_ret_pair()
// LLVM:   %[[RES:.*]] = call [2 x i32] @ret_pair()
// LLVM:   store [2 x i32] %[[RES]], ptr addrspace(5) %{{.*}}, align 4

// A _Complex up to 64 bits is packed like a small aggregate.
_Complex float ret_cfloat(void) { return 0; }

// CIR: cir.func {{.*}}@ret_cfloat() -> !cir.array<!u32i x 2>
// CIR:   %[[RET_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!cir.complex<!cir.float>, target_address_space(5)> -> !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>, !cir.array<!u32i x 2>
// CIR-NEXT:   cir.return %[[RET]] : !cir.array<!u32i x 2>
// LLVM: define {{.*}}[2 x i32] @ret_cfloat()
// LLVM:   %[[RET:.*]] = load [2 x i32], ptr addrspace(5) %{{.*}}, align 4
// LLVM-NEXT:   ret [2 x i32] %[[RET]]

// A wider _Complex that fits the register budget returns whole.
_Complex double ret_cdouble(void) { return 0; }

// CIR: cir.func {{.*}}@ret_cdouble() -> ![[PAIR:rec_anon_struct[0-9]*]]
// CIR:   %[[RET_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)> -> !cir.ptr<![[PAIR]], target_address_space(5)>
// CIR-NEXT:   %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<![[PAIR]], target_address_space(5)>, ![[PAIR]]
// CIR-NEXT:   cir.return %[[RET]] : ![[PAIR]]
// LLVM: define {{.*}}{ double, double } @ret_cdouble()
// LLVM:   ret { double, double } %{{.*}}

// An aggregate past the register budget returns through sret in addrspace(5).
big_t ret_big(void) {
  big_t b = {0};
  return b;
}

// CIR: cir.func {{.*}}@ret_big(%[[RET:[^:]+]]: !cir.ptr<!rec_big_t, target_address_space(5)> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_big_t, llvm.writable}
// CIR-NEXT:   %[[RET_VIEW:.*]] = cir.cast address_space %[[RET]] : !cir.ptr<!rec_big_t, target_address_space(5)> -> !cir.ptr<!rec_big_t>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_big_t
// CIR-NEXT:   cir.store align(4) %[[ZERO]], %[[RET_VIEW]] : !rec_big_t, !cir.ptr<!rec_big_t>
// CIR-NEXT:   cir.return
// LLVM: define {{.*}}void @ret_big(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.big_t) align 4 %[[RET:[^)]+]])
// LLVM:   %{{.*}} = addrspacecast ptr addrspace(5) %[[RET]] to ptr

// The caller's local is passed as the sret.
void call_ret_big(void) { big_t b = ret_big(); }

// CIR: cir.func {{.*}}@call_ret_big()
// CIR-NEXT:   %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!rec_big_t, target_address_space(5)>
// CIR:   cir.call @ret_big(%[[B]]) {{.*}}: (!cir.ptr<!rec_big_t, target_address_space(5)> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.sret = !rec_big_t, llvm.writable}) -> ()
// LLVM: define {{.*}}void @call_ret_big()
// LLVM:   alloca %struct.big_t, align 4, addrspace(5)
// LLVM:   call void @ret_big(ptr addrspace(5) dead_on_unwind writable sret(%struct.big_t) align 4 %{{.*}})

// A struct with a flexible array member returns through sret in addrspace(5).
fam_t ret_fam(void) {
  fam_t s = {0};
  return s;
}

// CIR: cir.func {{.*}}@ret_fam(%[[RET:[^:]+]]: !cir.ptr<!rec_fam_t, target_address_space(5)> {llvm.align = 4 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_fam_t, llvm.writable}
// CIR-NEXT:   %[[RET_VIEW:.*]] = cir.cast address_space %[[RET]] : !cir.ptr<!rec_fam_t, target_address_space(5)> -> !cir.ptr<!rec_fam_t>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.zero : !rec_fam_t
// CIR-NEXT:   cir.store align(4) %[[ZERO]], %[[RET_VIEW]] : !rec_fam_t, !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.return
// LLVM: define {{.*}}void @ret_fam(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.fam_t) align 4 %[[RET:[^)]+]])
// LLVM:   %{{.*}} = addrspacecast ptr addrspace(5) %[[RET]] to ptr

// A union return is not flattened.
u_pad_t ret_u_pad(u_pad_t *p) { return *p; }

// CIR: cir.func {{.*}}@ret_u_pad(%{{.*}}: !cir.ptr<!rec_u_pad_t> {llvm.noundef}{{.*}}) -> !rec_u_pad_t
// CIR:   %[[RET:.*]] = cir.load %{{.*}} : !cir.ptr<!rec_u_pad_t, target_address_space(5)>, !rec_u_pad_t
// CIR-NEXT:   cir.return %[[RET]] : !rec_u_pad_t
// LLVM: define {{.*}}%union.u_pad_t @ret_u_pad(ptr noundef %{{.*}})
// LLVM:   %[[RET:.*]] = load %union.u_pad_t, ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   ret %union.u_pad_t %[[RET]]

// A one-field struct return is not unwrapped.
cdwrap_t ret_cdwrap(void) {
  cdwrap_t w = {0};
  return w;
}

// CIR: cir.func {{.*}}@ret_cdwrap() -> !rec_cdwrap_t
// CIR:   %[[RET:.*]] = cir.load %{{.*}} : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>, !rec_cdwrap_t
// CIR-NEXT:   cir.return %[[RET]] : !rec_cdwrap_t
// LLVM: define {{.*}}%struct.cdwrap_t @ret_cdwrap()
// LLVM:   %[[RET:.*]] = load %struct.cdwrap_t, ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   ret %struct.cdwrap_t %[[RET]]
