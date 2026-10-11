// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// Check that the ABI library classifies return types the same way Clang does.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { char a, b, c; } small_t;
typedef struct { int a, b; } pair_t;
typedef struct { int a, b, c, d; } quad_t;
typedef struct { int a[4]; } array_wrapper_t;
typedef struct { int a[64]; } large_t;
typedef __attribute__((ext_vector_type(3))) char char3;
typedef struct { _Bool b; } bool_wrapper_t;
typedef struct { _BitInt(24) i; } bitint24_wrapper_t;

void ret_void(void) {}
// CHECK: define{{.*}} void @ret_void()

_Bool ret_bool(void) { return 0; }
// CHECK: define{{.*}} zeroext i1 @ret_bool()

char ret_char(void) { return 0; }
// CHECK: define{{.*}} signext i8 @ret_char()

int ret_int(void) { return 0; }
// CHECK: define{{.*}} i32 @ret_int()

long ret_long(void) { return 0; }
// CHECK: define{{.*}} i64 @ret_long()

float ret_float(void) { return 0; }
// CHECK: define{{.*}} float @ret_float()

_BitInt(65) ret_bitint65(void) { return 0; }
// CHECK: define{{.*}} i65 @ret_bitint65()

_BitInt(129) ret_bitint129(void) { return 0; }
// GCN: define{{.*}} void @ret_bitint129(ptr addrspace(5) dead_on_unwind noalias writable sret(i192) align 8 %{{.*}})
// SPIRV: define{{.*}} void @ret_bitint129(ptr dead_on_unwind noalias writable sret(i192) align 8 %{{.*}})

empty_t ret_empty(void) { empty_t e; return e; }
// CHECK: define{{.*}} void @ret_empty()

i8_wrapper_t ret_i8_wrapper(void) { i8_wrapper_t s; return s; }
// CHECK: define{{.*}} i8 @ret_i8_wrapper()

f32_wrapper_t ret_f32_wrapper(void) { f32_wrapper_t s; return s; }
// CHECK: define{{.*}} float @ret_f32_wrapper()

// Compared at in-memory size, so an element narrower than its storage unwraps.
bool_wrapper_t ret_bool_wrapper(void) { bool_wrapper_t s; return s; }
// CHECK: define{{.*}} i1 @ret_bool_wrapper()

bitint24_wrapper_t ret_bitint24_wrapper(void) { bitint24_wrapper_t s; return s; }
// CHECK: define{{.*}} i24 @ret_bitint24_wrapper()

small_t ret_small(void) { small_t s; return s; }
// CHECK: define{{.*}} i32 @ret_small()

pair_t ret_pair(void) { pair_t s; return s; }
// CHECK: define{{.*}} [2 x i32] @ret_pair()

// Over 8 bytes but within the register budget, so still returned directly.
quad_t ret_quad(void) { quad_t s; return s; }
// CHECK: define{{.*}} %struct.quad_t @ret_quad()

array_wrapper_t ret_array_wrapper(void) { array_wrapper_t s; return s; }
// CHECK: define{{.*}} %struct.array_wrapper_t @ret_array_wrapper()

large_t ret_large(void) { large_t s; return s; }
// GCN: define{{.*}} void @ret_large(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.large_t) align 4 %{{.*}})
// SPIRV: define{{.*}} void @ret_large(ptr dead_on_unwind noalias writable sret(%struct.large_t) align 4 %{{.*}})

char3 ret_char3(void) { char3 v; return v; }
// CHECK: define{{.*}} <3 x i8> @ret_char3()
