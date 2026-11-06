// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GCN
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-max-bitint-width=1024 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -fexperimental-max-bitint-width=1024 -fexperimental-abi-lowering -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// Check that the ABI library classifies arguments the same way Clang does.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { int a[4]; } array_wrapper_t;
typedef struct { char a, b, c; } small_t;
typedef struct { int a, b, c, d; } pair16_t;
typedef struct { int a[64]; } large_t;
typedef struct { int i; float f; double d; } mixed_t;
typedef union __attribute__((transparent_union)) { int i; } transparent_u;
typedef __attribute__((ext_vector_type(3))) char char3;
typedef __attribute__((ext_vector_type(4))) short short4;
typedef struct { _Bool b; } bool_wrapper_t;
typedef struct { _BitInt(24) i; } bitint24_wrapper_t;
typedef struct { char3 v; } char3_wrapper_t;
typedef union { long long a; char c[]; } fam_union_t;

void arg_void(void) {}
// CHECK: define{{.*}} void @arg_void()

void arg_bool(_Bool b) {}
// CHECK: define{{.*}} void @arg_bool(i1 noundef zeroext %{{.*}})

void arg_char(char c) {}
// CHECK: define{{.*}} void @arg_char(i8 noundef signext %{{.*}})

void arg_short(short s) {}
// CHECK: define{{.*}} void @arg_short(i16 noundef signext %{{.*}})

void arg_int(int i) {}
// CHECK: define{{.*}} void @arg_int(i32 noundef %{{.*}})

void arg_long(long l) {}
// CHECK: define{{.*}} void @arg_long(i64 noundef %{{.*}})

void arg_float(float f) {}
// CHECK: define{{.*}} void @arg_float(float noundef %{{.*}})

void arg_double(double d) {}
// CHECK: define{{.*}} void @arg_double(double noundef %{{.*}})

void arg_ptr(int *p) {}
// GCN: define{{.*}} void @arg_ptr(ptr noundef %{{.*}})
// SPIRV: define{{.*}} void @arg_ptr(ptr addrspace(4) noundef %{{.*}})

void arg_bitint65(_BitInt(65) b) {}
// CHECK: define{{.*}} void @arg_bitint65(i65 noundef %{{.*}})

void arg_bitint129(_BitInt(129) b) {}
// GCN: define{{.*}} void @arg_bitint129(ptr addrspace(5) noundef byval(i192) align 8 %{{.*}})
// SPIRV: define{{.*}} void @arg_bitint129(ptr noundef byval(i192) align 8 %{{.*}})

void arg_empty(empty_t e) {}
// CHECK: define{{.*}} void @arg_empty()

void arg_i8_wrapper(i8_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_i8_wrapper(i8 %{{.*}})

void arg_i32_wrapper(i32_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_i32_wrapper(i32 %{{.*}})

void arg_f32_wrapper(f32_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_f32_wrapper(float %{{.*}})

// Compared at in-memory size, so an element narrower than its storage unwraps.
void arg_bool_wrapper(bool_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_bool_wrapper(i1 %{{.*}})

void arg_bitint24_wrapper(bitint24_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_bitint24_wrapper(i24 %{{.*}})

void arg_char3_wrapper(char3_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_char3_wrapper(<3 x i8> %{{.*}})

void arg_fam_union(fam_union_t u) {}
// GCN: define{{.*}} void @arg_fam_union(ptr addrspace(5) noundef byval(%union.fam_union_t) align 8 %{{.*}})
// SPIRV: define{{.*}} void @arg_fam_union(ptr noundef byval(%union.fam_union_t) align 8 %{{.*}})

// Not a single element, so this falls through to register packing.
void arg_array_wrapper(array_wrapper_t s) {}
// CHECK: define{{.*}} void @arg_array_wrapper([4 x i32] %{{.*}})

void arg_small(small_t s) {}
// CHECK: define{{.*}} void @arg_small(i32 %{{.*}})

void arg_pair16(pair16_t s) {}
// CHECK: define{{.*}} void @arg_pair16(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

void arg_large(large_t s) {}
// GCN: define{{.*}} void @arg_large(ptr addrspace(5) noundef byref(%struct.large_t) align 4 %{{.*}})
// SPIRV: define{{.*}} void @arg_large(ptr noundef byref(%struct.large_t) align 4 %{{.*}})

void arg_mixed(mixed_t s) {}
// CHECK: define{{.*}} void @arg_mixed(i32 %{{.*}}, float %{{.*}}, double %{{.*}})

void arg_transparent_union(transparent_u u) {}
// CHECK: define{{.*}} void @arg_transparent_union(i32 %{{.*}})

void arg_char3(char3 v) {}
// CHECK: define{{.*}} void @arg_char3(<3 x i8> noundef %{{.*}})

void arg_short4(short4 v) {}
// CHECK: define{{.*}} void @arg_short4(<4 x i16> noundef %{{.*}})

void arg_variadic(int n, ...) {}
// CHECK: define{{.*}} void @arg_variadic(i32 noundef %{{.*}}, ...)

// A variadic argument is passed as-is, not packed like a fixed one.
void call_variadic(small_t s) { arg_variadic(1, s); }
// CHECK: define{{.*}} void @call_variadic(i32 %{{.*}})
// CHECK: call{{.*}} void (i32, ...){{.*}} @arg_variadic(i32 noundef 1, %struct.small_t %{{.*}})

// The 16-register budget is shared, so the fifth aggregate goes by reference.
void arg_reg_budget(pair16_t a, pair16_t b, pair16_t c, pair16_t d, pair16_t e) {}
// GCN: define{{.*}} void @arg_reg_budget(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr addrspace(5) noundef byref(%struct.pair16_t) align 4 %{{.*}})
// SPIRV: define{{.*}} void @arg_reg_budget(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, ptr noundef byref(%struct.pair16_t) align 4 %{{.*}})

// A 16-bit element vector packs two per register, so six short4 cost 12 of 16.
void arg_packed_short4_budget(short4 a, short4 b, short4 c, short4 d, short4 e,
                              short4 f, pair16_t g) {}
// CHECK: define{{.*}} void @arg_packed_short4_budget(<4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}}, <4 x i16> noundef %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
