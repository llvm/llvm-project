// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -emit-llvm %s -o - | FileCheck %s %}

// Checks how non-kernel AMDGPU functions pass arguments.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { int *p; } ptr_wrapper_t;
typedef struct { char a, b; } char2_t;
typedef struct { char a, b, c; } small_t;
typedef union { int i; float f; } int_float_t;
typedef union {
  int *ip;
  float *fp;
} __attribute__((transparent_union)) transparent_t;
typedef struct { int a, b; } pair_t;
typedef struct { int a, b, c; } triple_t;
typedef struct { int data[17]; } big_t;
typedef struct { int data[16]; } big16_t;
typedef struct { int a : 3; int b : 5; } bits_t;
typedef struct { int n; int data[]; } fam_t;
typedef struct { _Complex double c; } cdwrap_t;
typedef struct { int v[3]; } arrwrap_t;
typedef struct { struct { int a, b, c; } t; } nestwrap_t;
typedef union { int a[3]; double d; } u_pad_t;
typedef union { double d[2]; long l[2]; } u_dl_t;
typedef union __attribute__((packed)) { char c[13]; double d; } u_packed_t;
typedef short short4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));

void arg_void(void) {}

// CHECK: define {{.*}}void @arg_void()

// Sub-word integers and bool carry signext or zeroext per their signedness.
void arg_bool(_Bool b) {}

// CHECK: define {{.*}}void @arg_bool(i1 noundef zeroext %{{.*}})

void arg_char(char c) {}

// CHECK: define {{.*}}void @arg_char(i8 noundef signext %{{.*}})

void arg_uchar(unsigned char c) {}

// CHECK: define {{.*}}void @arg_uchar(i8 noundef zeroext %{{.*}})

void arg_short(short s) {}

// CHECK: define {{.*}}void @arg_short(i16 noundef signext %{{.*}})

// The call site carries the same extension as the callee.
void call_char(char c) { arg_char(c); }

// CHECK: define {{.*}}void @call_char(i8 noundef signext %{{.*}})
// CHECK:   call void @arg_char(i8 noundef signext %{{.*}})

// Word-sized and wider scalars pass through unchanged.
void arg_int(int i) {}

// CHECK: define {{.*}}void @arg_int(i32 noundef %{{.*}})

void arg_long(long l) {}

// CHECK: define {{.*}}void @arg_long(i64 noundef %{{.*}})

// 128-bit integers pass whole.
void arg_int128(__int128 i) {}

// CHECK: define {{.*}}void @arg_int128(i128 noundef %[[I:[^)]+]])
// CHECK:   store i128 %[[I]], ptr %{{.*}}

void arg_bitint128(_BitInt(128) i) {}

// CHECK: define {{.*}}void @arg_bitint128(i128 noundef %[[I:[^)]+]])
// CHECK:   store i128 %[[I]], ptr %{{.*}}, align 8

void arg_float(float f) {}

// CHECK: define {{.*}}void @arg_float(float noundef %{{.*}})

void arg_double(double d) {}

// CHECK: define {{.*}}void @arg_double(double noundef %{{.*}})

void arg_half(_Float16 h) {}

// CHECK: define {{.*}}void @arg_half(half noundef %{{.*}})

// Outside a HIP kernel a generic pointer stays generic.
void arg_ptr(int *p) {}

// CHECK: define {{.*}}void @arg_ptr(ptr noundef %{{.*}})

// A _BitInt wider than 64 bits and vectors, including 16-bit and 3-element
// ones, pass whole.
void arg_bitint65(_BitInt(65) i) {}

// CHECK: define {{.*}}void @arg_bitint65(i65 noundef %{{.*}})

void arg_short4(short4 v) {}

// CHECK: define {{.*}}void @arg_short4(<4 x i16> noundef %{{.*}})

void arg_float3(float3 v) {}

// CHECK: define {{.*}}void @arg_float3(<3 x float> noundef %{{.*}})

// An empty struct is dropped from the signature.
void arg_empty(empty_t e) {}

// CHECK: define {{.*}}void @arg_empty()

// Single-element structs pass as their element. The callee rebuilds the
// struct in a private (addrspace 5) slot.
void arg_i8_wrapper(i8_wrapper_t w) {}

// CHECK: define {{.*}}void @arg_i8_wrapper(i8 %{{.*}})

void arg_i32_wrapper(i32_wrapper_t w) {}

// CHECK: define {{.*}}void @arg_i32_wrapper(i32 %{{.*}})

void arg_f32_wrapper(f32_wrapper_t w) {}

// CHECK: define {{.*}}void @arg_f32_wrapper(float %{{.*}})

void arg_ptr_wrapper(ptr_wrapper_t w) {}

// CHECK: define {{.*}}void @arg_ptr_wrapper(ptr %{{.*}})

// A struct of one aggregate field is passed as that field.
void arg_cdwrap(cdwrap_t w) {}

// CHECK: define {{.*}}void @arg_cdwrap({ double, double } %{{.*}})

void arg_arrwrap(arrwrap_t w) {}

// CHECK: define {{.*}}void @arg_arrwrap([3 x i32] %[[W:[^)]+]])
// CHECK:   %[[W_V:.*]] = getelementptr inbounds nuw %struct.arrwrap_t, ptr addrspace(5) %{{.*}}, i32 0, i32 0
// CHECK-NEXT:   store [3 x i32] %[[W]], ptr addrspace(5) %[[W_V]], align 4

void arg_nestwrap(nestwrap_t w) {}

// CHECK: define {{.*}}void @arg_nestwrap(%struct.anon{{.*}} %{{.*}})

// Aggregates up to 32 bits are packed into one integer register.
void arg_char2(char2_t s) {}

// CHECK: define {{.*}}void @arg_char2(i16 %{{.*}})

void arg_small(small_t s) {}

// CHECK: define {{.*}}void @arg_small(i32 %{{.*}})

void arg_union(int_float_t u) {}

// CHECK: define {{.*}}void @arg_union(i32 %{{.*}})

// Bit-fields are packed like any other small aggregate.
void arg_bits(bits_t b) {}

// CHECK: define {{.*}}void @arg_bits(i32 %[[B:[^)]+]])
// CHECK:   store i32 %[[B]], ptr addrspace(5) %{{.*}}, align 4

// Aggregates of 33 to 64 bits are packed into a pair of i32 registers.
void arg_pair(pair_t p) {}

// CHECK: define {{.*}}void @arg_pair([2 x i32] %[[P:[^)]+]])
// CHECK:   store [2 x i32] %[[P]], ptr addrspace(5) %{{.*}}, align 4

// A _Complex up to 64 bits is packed like a small aggregate.
void arg_cfloat(_Complex float c) {}

// CHECK: define {{.*}}void @arg_cfloat([2 x i32] noundef %[[C:[^)]+]])
// CHECK:   store [2 x i32] %[[C]], ptr addrspace(5) %{{.*}}, align 4

// The call site is coerced the same way as the callee.
void call_small(small_t s) { arg_small(s); }

// CHECK: define {{.*}}void @call_small(i32 %{{.*}})
// CHECK:   call void @arg_small(i32 %{{.*}})

void call_pair(pair_t p) { arg_pair(p); }

// CHECK: define {{.*}}void @call_pair(
// CHECK:   %[[ARG:.*]] = load [2 x i32], ptr %{{.*}}, align 4
// CHECK:   call void @arg_pair([2 x i32] %[[ARG]])

// A transparent union is passed as its first member.
void arg_transparent(transparent_t u) {}

// CHECK: define {{.*}}void @arg_transparent(ptr %[[U:[^)]+]])
// CHECK:   store ptr %[[U]], ptr addrspace(5) %{{.*}}, align 8

// A caller can pass any member type directly.
void call_transparent(int *p) { arg_transparent(p); }

// CHECK: define {{.*}}void @call_transparent(ptr noundef %{{.*}})
// CHECK:   call void @arg_transparent(ptr %{{.*}})

// Larger aggregates that fit the register budget pass one field per register.
void arg_triple(triple_t t) {}

// CHECK: define {{.*}}void @arg_triple(i32 %[[T0:[^,]+]], i32 %[[T1:[^,]+]], i32 %[[T2:[^)]+]])
// CHECK:   store i32 %[[T0]], ptr addrspace(5) %{{.*}}, align 4
// CHECK:   store i32 %[[T1]], ptr addrspace(5) %{{.*}}, align 4
// CHECK:   store i32 %[[T2]], ptr addrspace(5) %{{.*}}, align 4

void call_triple(triple_t t) { arg_triple(t); }

// CHECK: define {{.*}}void @call_triple(
// CHECK:   call void @arg_triple(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

// A call through a function pointer is coerced like a direct call.
void call_fnptr(void (*fp)(triple_t), triple_t t) { fp(t); }

// CHECK: define {{.*}}void @call_fnptr(ptr noundef %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
// CHECK:   %[[FN:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK:   call void %[[FN]](i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

// A _Complex wider than 64 bits is split into its two parts.
void arg_cdouble(_Complex double c) {}

// CHECK: define {{.*}}void @arg_cdouble(double noundef %[[RE:[^,]+]], double noundef %[[IM:[^)]+]])
// CHECK:   %[[C_RE:.*]] = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %[[C:.*]], i32 0, i32 0
// CHECK-NEXT:   store double %[[RE]], ptr addrspace(5) %[[C_RE]], align 8
// CHECK-NEXT:   %[[C_IM:.*]] = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %[[C]], i32 0, i32 1
// CHECK-NEXT:   store double %[[IM]], ptr addrspace(5) %[[C_IM]], align 8

void arg_clong(_Complex long c) {}

// CHECK: define {{.*}}void @arg_clong(i64 noundef %[[RE:[^,]+]], i64 noundef %[[IM:[^)]+]])
// CHECK:   store i64 %[[RE]], ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw { i64, i64 }, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   store i64 %[[IM]], ptr addrspace(5) %{{.*}}, align 8

// The caller splits it the same way.
void call_cdouble(_Complex double c) { arg_cdouble(c); }

// CHECK: define {{.*}}void @call_cdouble(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK:   %[[RE:.*]] = load double, ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   %[[IM:.*]] = load double, ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   call void @arg_cdouble(double noundef %[[RE]], double noundef %[[IM]])

// A union is passed as its storage member, then its tail padding.
void arg_u_pad(u_pad_t u) {}

// CHECK: define {{.*}}void @arg_u_pad(double %[[D:[^,]+]], [8 x i8] %[[PAD:[^)]+]])
// CHECK:   store double %[[D]], ptr addrspace(5) %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   store [8 x i8] %[[PAD]], ptr addrspace(5) %{{.*}}

void call_u_pad(u_pad_t u) { arg_u_pad(u); }

// CHECK: define {{.*}}void @call_u_pad(double %{{.*}}, [8 x i8] %{{.*}})
// CHECK:   %[[D:.*]] = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   %[[PAD:.*]] = load [8 x i8], ptr %{{.*}}, align 8
// CHECK-NEXT:   call void @arg_u_pad(double %[[D]], [8 x i8] %[[PAD]])

// With no tail padding a union is passed as its storage member alone.
void arg_u_dl(u_dl_t u) {}

// CHECK: define {{.*}}void @arg_u_dl([2 x double] %[[U:[^)]+]])
// CHECK:   store [2 x double] %[[U]], ptr addrspace(5) %{{.*}}, align 8

// A packed union keeps its padding, and its double is stored at align 1.
void arg_u_packed(u_packed_t u) {}

// CHECK: define {{.*}}void @arg_u_packed(double %[[D:[^,]+]], [5 x i8] %[[PAD:[^)]+]])
// CHECK:   store double %[[D]], ptr addrspace(5) %{{.*}}, align 1
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   store [5 x i8] %[[PAD]], ptr addrspace(5) %{{.*}}, align 1

// An aggregate of exactly 16 registers is passed direct.
void arg_big16(big16_t b) {}

// CHECK: define {{.*}}void @arg_big16([16 x i32] %[[B:[^)]+]])
// CHECK:   %[[B_DATA:.*]] = getelementptr inbounds nuw %struct.big16_t, ptr addrspace(5) %{{.*}}, i32 0, i32 0
// CHECK-NEXT:   store [16 x i32] %[[B]], ptr addrspace(5) %[[B_DATA]], align 4

// An aggregate past the register budget is passed byref in addrspace(5).
void arg_big(big_t b) {}

// CHECK: define {{.*}}void @arg_big(ptr addrspace(5) noundef byref(%struct.big_t) align 4 %[[B_ARG:.*]])
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(5) align 4 %[[B_ARG]], i64 68, i1 false)

// The caller passes its own local, not a copy of it.
void call_big(void) {
  big_t b = {0};
  arg_big(b);
}

// CHECK: define {{.*}}void @call_big()
// CHECK:   call void @arg_big(ptr addrspace(5) noundef byref(%struct.big_t) align 4 %{{.*}})

// Scalars use the same register budget, so a struct that would fit on its
// own is passed byref once they have used it up.
void arg_budget(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                int a7, int a8, int a9, int a10, int a11, int a12, int a13,
                triple_t t) {}

// CHECK: define {{.*}}void @arg_budget(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr addrspace(5) noundef byref(%struct.triple_t) align 4 %[[T_ARG:.*]])
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(5) align 4 %[[T_ARG]], i64 12, i1 false)

// A struct that takes exactly the registers left is still passed direct.
void arg_budget_exact(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                      int a7, int a8, int a9, int a10, int a11, int a12,
                      triple_t t) {}

// CHECK: define {{.*}}void @arg_budget_exact(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 %[[T0:[^,]+]], i32 %[[T1:[^,]+]], i32 %[[T2:[^)]+]])
// CHECK:   store i32 %[[T0]], ptr addrspace(5) %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw %struct.triple_t, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// CHECK-NEXT:   store i32 %[[T1]], ptr addrspace(5) %{{.*}}, align 4
// CHECK-NEXT:   %{{.*}} = getelementptr inbounds nuw %struct.triple_t, ptr addrspace(5) %{{.*}}, i32 0, i32 2
// CHECK-NEXT:   store i32 %[[T2]], ptr addrspace(5) %{{.*}}, align 4

// An aggregate of up to 64 bits is packed even once the budget is used up.
void arg_budget_pair(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                     int a7, int a8, int a9, int a10, int a11, int a12,
                     int a13, int a14, int a15, pair_t p) {}

// CHECK: define {{.*}}void @arg_budget_pair(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, [2 x i32] %[[P:[^)]+]])
// CHECK:   store [2 x i32] %[[P]], ptr addrspace(5) %{{.*}}, align 4

// A packed aggregate still uses budget, so the next struct no longer fits.
void arg_budget_pair_triple(int a0, int a1, int a2, int a3, int a4, int a5,
                            int a6, int a7, int a8, int a9, int a10, int a11,
                            int a12, int a13, pair_t p, triple_t t) {}

// CHECK: define {{.*}}void @arg_budget_pair_triple(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, [2 x i32] %[[P:[^,]+]], ptr addrspace(5) noundef byref(%struct.triple_t) align 4 %[[T_ARG:[^)]+]])
// CHECK:   store [2 x i32] %[[P]], ptr addrspace(5) %{{.*}}, align 4
// CHECK:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(5) align 4 %[[T_ARG]], i64 12, i1 false)

// Past the register budget a _Complex is passed byref in addrspace(5).
void arg_cdouble_budget(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                        int a7, int a8, int a9, int a10, int a11, int a12,
                        int a13, _Complex double c) {}

// CHECK: define {{.*}}void @arg_cdouble_budget(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr addrspace(5) noundef byref({ double, double }) align 8 %{{.*}})

// A struct with a flexible array member is passed byval in addrspace(5).
void arg_fam(fam_t s) {}

// CHECK: define {{.*}}void @arg_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})

// The caller passes its own local as the byval pointer.
void call_fam(fam_t s) { arg_fam(s); }

// CHECK: define {{.*}}void @call_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})
// CHECK:   call void @arg_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})

void take_var(int n, ...);

// Promoted scalars pass through an ellipsis unchanged.
void call_var_scalars(char c, double d) { take_var(2, c, d); }

// CHECK: define {{.*}}void @call_var_scalars(
// CHECK:   %[[C_INT:.*]] = sext i8 %{{.*}} to i32
// CHECK-NEXT:   %[[D:.*]] = load double, ptr %{{.*}}, align 8
// CHECK-NEXT:   call void (i32, ...) @take_var(i32 noundef 2, i32 noundef %[[C_INT]], double noundef %[[D]])
// CHECK: declare void @take_var(i32 noundef, ...)

// Aggregates pass through an ellipsis whole, not packed into registers.
void call_var_aggs(pair_t p, small_t s) { take_var(2, p, s); }

// CHECK: define {{.*}}void @call_var_aggs([2 x i32] %{{.*}}, i32 %{{.*}})
// CHECK:   %[[P:.*]] = load %struct.pair_t, ptr %{{.*}}, align 4
// CHECK-NEXT:   %[[S:.*]] = load %struct.small_t, ptr %{{.*}}, align 1
// CHECK-NEXT:   call void (i32, ...) @take_var(i32 noundef 2, %struct.pair_t %[[P]], %struct.small_t %[[S]])

// Through an ellipsis a _Complex is passed whole.
void call_var_cdouble(_Complex double c) { take_var(1, c); }

// CHECK: define {{.*}}void @call_var_cdouble(double noundef %{{.*}}, double noundef %{{.*}})
// CHECK:   call void (i32, ...) @take_var(i32 noundef 1, { double, double } noundef %{{.*}})
