// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies arguments of ordinary AMDGPU
// functions, which use the default convention rather than the kernel one.

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

// CIR: cir.func {{.*}}@arg_void()
// LLVM: define {{.*}}void @arg_void()

// Sub-word integers and bool carry signext or zeroext per their signedness.
void arg_bool(_Bool b) {}

// CIR: cir.func {{.*}}@arg_bool(%arg0: !cir.bool {{.*}}llvm.zeroext
// LLVM: define {{.*}}void @arg_bool(i1 noundef zeroext %{{.*}})

void arg_char(char c) {}

// CIR: cir.func {{.*}}@arg_char(%arg0: !s8i {{.*}}llvm.signext
// LLVM: define {{.*}}void @arg_char(i8 noundef signext %{{.*}})

void arg_uchar(unsigned char c) {}

// CIR: cir.func {{.*}}@arg_uchar(%arg0: !u8i {{.*}}llvm.zeroext
// LLVM: define {{.*}}void @arg_uchar(i8 noundef zeroext %{{.*}})

void arg_short(short s) {}

// CIR: cir.func {{.*}}@arg_short(%arg0: !s16i {{.*}}llvm.signext
// LLVM: define {{.*}}void @arg_short(i16 noundef signext %{{.*}})

// Word-sized and wider scalars pass through unchanged.
void arg_int(int i) {}

// CIR: cir.func {{.*}}@arg_int(%arg0: !s32i {{.*}})
// LLVM: define {{.*}}void @arg_int(i32 noundef %{{.*}})

void arg_long(long l) {}

// CIR: cir.func {{.*}}@arg_long(%arg0: !s64i {{.*}})
// LLVM: define {{.*}}void @arg_long(i64 noundef %{{.*}})

void arg_float(float f) {}

// CIR: cir.func {{.*}}@arg_float(%arg0: !cir.float {{.*}})
// LLVM: define {{.*}}void @arg_float(float noundef %{{.*}})

void arg_double(double d) {}

// CIR: cir.func {{.*}}@arg_double(%arg0: !cir.double {{.*}})
// LLVM: define {{.*}}void @arg_double(double noundef %{{.*}})

void arg_half(_Float16 h) {}

// CIR: cir.func {{.*}}@arg_half(%arg0: !cir.f16 {{.*}})
// LLVM: define {{.*}}void @arg_half(half noundef %{{.*}})

// Outside a HIP kernel a generic pointer stays generic.
void arg_ptr(int *p) {}

// CIR: cir.func {{.*}}@arg_ptr(%arg0: !cir.ptr<!s32i> {{.*}})
// LLVM: define {{.*}}void @arg_ptr(ptr noundef %{{.*}})

// A _BitInt wider than 64 bits and vectors, including 16-bit and 3-element
// ones, pass whole.
void arg_bitint65(_BitInt(65) i) {}

// CIR: cir.func {{.*}}@arg_bitint65(%arg0: !cir.int<s, 65, bitint> {{.*}})
// LLVM: define {{.*}}void @arg_bitint65(i65 noundef %{{.*}})

void arg_short4(short4 v) {}

// CIR: cir.func {{.*}}@arg_short4(%arg0: !cir.vector<4 x !s16i> {{.*}})
// LLVM: define {{.*}}void @arg_short4(<4 x i16> noundef %{{.*}})

void arg_float3(float3 v) {}

// CIR: cir.func {{.*}}@arg_float3(%arg0: !cir.vector<3 x !cir.float> {{.*}})
// LLVM: define {{.*}}void @arg_float3(<3 x float> noundef %{{.*}})

// An empty struct is dropped from the signature.
void arg_empty(empty_t e) {}

// CIR: cir.func {{.*}}@arg_empty()
// LLVM: define {{.*}}void @arg_empty()

// Single-element structs pass as their element. The callee rebuilds the
// struct in a private (addrspace 5) slot.
void arg_i8_wrapper(i8_wrapper_t w) {}

// CIR: cir.func {{.*}}@arg_i8_wrapper(%arg0: !s8i{{.*}})
// CIR:   cir.alloca "coerce" {{.*}} : !cir.ptr<{{.*}}, target_address_space(5)>
// LLVM: define {{.*}}void @arg_i8_wrapper(i8 %{{.*}})

void arg_i32_wrapper(i32_wrapper_t w) {}

// CIR: cir.func {{.*}}@arg_i32_wrapper(%arg0: !s32i{{.*}})
// LLVM: define {{.*}}void @arg_i32_wrapper(i32 %{{.*}})

void arg_f32_wrapper(f32_wrapper_t w) {}

// CIR: cir.func {{.*}}@arg_f32_wrapper(%arg0: !cir.float{{.*}})
// LLVM: define {{.*}}void @arg_f32_wrapper(float %{{.*}})

void arg_ptr_wrapper(ptr_wrapper_t w) {}

// CIR: cir.func {{.*}}@arg_ptr_wrapper(%arg0: !cir.ptr<{{[^,]*}}>{{.*}})
// LLVM: define {{.*}}void @arg_ptr_wrapper(ptr %{{.*}})

// Aggregates up to 32 bits are packed into one integer register.
void arg_char2(char2_t s) {}

// CIR: cir.func {{.*}}@arg_char2(%arg0: !u16i{{.*}})
// LLVM: define {{.*}}void @arg_char2(i16 %{{.*}})

void arg_small(small_t s) {}

// CIR: cir.func {{.*}}@arg_small(%arg0: !u32i{{.*}})
// LLVM: define {{.*}}void @arg_small(i32 %{{.*}})

void arg_union(int_float_t u) {}

// CIR: cir.func {{.*}}@arg_union(%arg0: !u32i{{.*}})
// LLVM: define {{.*}}void @arg_union(i32 %{{.*}})

// A transparent union is passed as its first member.
void arg_transparent(transparent_t u) {}

// CIR: cir.func {{.*}}@arg_transparent(%[[U:[^:]+]]: !cir.ptr<!s32i>{{.*}})
// CIR-NEXT:   %[[U_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!cir.ptr<!s32i>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[U]], %[[U_COERCE]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_transparent(ptr %[[U:[^)]+]])
// LLVM:   store ptr %[[U]], ptr addrspace(5) %{{.*}}, align 8

// A caller can pass any member type directly.
void call_transparent(int *p) { arg_transparent(p); }

// CIR: cir.func {{.*}}@call_transparent(
// CIR:   %[[U_PTR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_transparent_t> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT:   %[[ARG:.*]] = cir.load align(8) %[[U_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR-NEXT:   cir.call @arg_transparent(%[[ARG]]) {{.*}}: (!cir.ptr<!s32i>) -> ()
// LLVM: define {{.*}}void @call_transparent(ptr noundef %{{.*}})
// LLVM:   call void @arg_transparent(ptr %{{.*}})

// Aggregates of 33 to 64 bits are packed into a pair of i32 registers.
void arg_pair(pair_t p) {}

// CIR: cir.func {{.*}}@arg_pair(%[[P:[^:]+]]: !cir.array<!u32i x 2>{{.*}})
// CIR-NEXT:   %[[P_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[P]], %[[P_COERCE]] : !cir.array<!u32i x 2>, !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_pair([2 x i32] %[[P:[^)]+]])
// LLVM:   store [2 x i32] %[[P]], ptr addrspace(5) %{{.*}}, align 4

// Larger aggregates that fit the register budget pass one field per register.
void arg_triple(triple_t t) {}

// CIR: cir.func {{.*}}@arg_triple(%[[T0:[^:]+]]: !s32i{{.*}}, %[[T1:[^:]+]]: !s32i{{.*}}, %[[T2:[^:]+]]: !s32i{{.*}})
// CIR-NEXT:   %[[T_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_triple_t, target_address_space(5)>
// CIR-NEXT:   %[[F0:.*]] = cir.get_member %[[T_COERCE]][0]
// CIR-NEXT:   cir.store %[[T0]], %[[F0]]
// CIR-NEXT:   %[[F1:.*]] = cir.get_member %[[T_COERCE]][1]
// CIR-NEXT:   cir.store %[[T1]], %[[F1]]
// CIR-NEXT:   %[[F2:.*]] = cir.get_member %[[T_COERCE]][2]
// CIR-NEXT:   cir.store %[[T2]], %[[F2]]
// LLVM: define {{.*}}void @arg_triple(i32 %[[T0:[^,]+]], i32 %[[T1:[^,]+]], i32 %[[T2:[^)]+]])
// LLVM:   store i32 %[[T0]], ptr addrspace(5) %{{.*}}, align 4
// LLVM:   store i32 %[[T1]], ptr addrspace(5) %{{.*}}, align 4
// LLVM:   store i32 %[[T2]], ptr addrspace(5) %{{.*}}, align 4

// The call site is coerced the same way as the callee.
void call_small(small_t s) { arg_small(s); }

// CIR: cir.func {{.*}}@call_small(%arg0: !u32i{{.*}})
// CIR:   cir.call @arg_small(%{{.*}}){{.*}}: (!u32i{{.*}}) -> ()
// LLVM: define {{.*}}void @call_small(i32 %{{.*}})
// LLVM:   call void @arg_small(i32 %{{.*}})

void call_pair(pair_t p) { arg_pair(p); }

// CIR: cir.func {{.*}}@call_pair(
// CIR:   %[[ARG:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!cir.array<!u32i x 2>>, !cir.array<!u32i x 2>
// CIR:   cir.call @arg_pair(%[[ARG]]) {{.*}}: (!cir.array<!u32i x 2>) -> ()
// LLVM: define {{.*}}void @call_pair(
// LLVM:   %[[ARG:.*]] = load [2 x i32], ptr %{{.*}}, align 4
// LLVM:   call void @arg_pair([2 x i32] %[[ARG]])

void call_triple(triple_t t) { arg_triple(t); }

// CIR: cir.func {{.*}}@call_triple(
// CIR:   cir.call @arg_triple(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!s32i, !s32i, !s32i) -> ()
// LLVM: define {{.*}}void @call_triple(
// LLVM:   call void @arg_triple(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

// The call site carries the same extension as the callee.
void call_char(char c) { arg_char(c); }

// LLVM: define {{.*}}void @call_char(i8 noundef signext %{{.*}})
// LLVM:   call void @arg_char(i8 noundef signext %{{.*}})

void take_var(int n, ...);

// Promoted scalars pass through an ellipsis unchanged.
void call_var_scalars(char c, double d) { take_var(2, c, d); }

// CIR: cir.func {{.*}}@call_var_scalars(
// CIR:   %[[C:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!s8i>, !s8i
// CIR-NEXT:   %[[C_INT:.*]] = cir.cast integral %[[C]] : !s8i -> !s32i
// CIR-NEXT:   %[[D:.*]] = cir.load align(8) %{{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:   cir.call @take_var(%{{.*}}, %[[C_INT]], %[[D]]) {{.*}}: (!s32i {llvm.noundef}, !s32i {llvm.noundef}, !cir.double {llvm.noundef}) -> ()
// CIR: cir.func private @take_var(!s32i {llvm.noundef}, ...)
// LLVM: define {{.*}}void @call_var_scalars(
// LLVM:   %[[C_INT:.*]] = sext i8 %{{.*}} to i32
// LLVM-NEXT:   %[[D:.*]] = load double, ptr %{{.*}}, align 8
// LLVM-NEXT:   call void (i32, ...) @take_var(i32 noundef 2, i32 noundef %[[C_INT]], double noundef %[[D]])
// LLVM: declare void @take_var(i32 noundef, ...)

// Aggregates pass through an ellipsis whole, not packed into registers.
void call_var_aggs(pair_t p, small_t s) { take_var(2, p, s); }

// CIR: cir.func {{.*}}@call_var_aggs(
// CIR:   %[[P:.*]] = cir.load align(4) %{{.*}} : !cir.ptr<!rec_pair_t>, !rec_pair_t
// CIR-NEXT:   %[[S:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!rec_small_t>, !rec_small_t
// CIR-NEXT:   cir.call @take_var(%{{.*}}, %[[P]], %[[S]]) {{.*}}: (!s32i {llvm.noundef}, !rec_pair_t, !rec_small_t) -> ()
// LLVM: define {{.*}}void @call_var_aggs([2 x i32] %{{.*}}, i32 %{{.*}})
// LLVM:   %[[P:.*]] = load %struct.pair_t, ptr %{{.*}}, align 4
// LLVM-NEXT:   %[[S:.*]] = load %struct.small_t, ptr %{{.*}}, align 1
// LLVM-NEXT:   call void (i32, ...) @take_var(i32 noundef 2, %struct.pair_t %[[P]], %struct.small_t %[[S]])

// An aggregate past the register budget is passed byref in addrspace(5).
void arg_big(big_t b) {}

// CIR: cir.func {{.*}}@arg_big(%[[B_ARG:[^:]+]]: !cir.ptr<!rec_big_t, target_address_space(5)> {{.*}}llvm.byref = !rec_big_t
// CIR:   cir.copy %[[B_ARG]] align(4) to %{{.*}} align(4) : !cir.ptr<!rec_big_t, target_address_space(5)>, !cir.ptr<!rec_big_t>
// LLVM: define {{.*}}void @arg_big(ptr addrspace(5) noundef byref(%struct.big_t) align 4 %[[B_ARG:.*]])
// LLVM:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(5) align 4 %[[B_ARG]], i64 68, i1 false)

// The caller passes its own local, not a copy of it.
void call_big(void) {
  big_t b = {0};
  arg_big(b);
}

// CIR: cir.func {{.*}}@call_big()
// CIR:   %[[B:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!rec_big_t, target_address_space(5)>
// CIR:   cir.call @arg_big(%[[B]]) {{.*}}: (!cir.ptr<!rec_big_t, target_address_space(5)> {{.*}}llvm.byref = !rec_big_t
// LLVM: define {{.*}}void @call_big()
// LLVM:   call void @arg_big(ptr addrspace(5) noundef byref(%struct.big_t) align 4 %{{.*}})

// Scalars use the same register budget, so a struct that would fit on its
// own is passed byref once they have used it up.
void arg_budget(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                int a7, int a8, int a9, int a10, int a11, int a12, int a13,
                triple_t t) {}

// CIR: cir.func {{.*}}@arg_budget(%arg0: !s32i {{.*}}%arg13: !s32i {{.*}}, %[[T_ARG:[^:]+]]: !cir.ptr<!rec_triple_t, target_address_space(5)> {{.*}}llvm.byref = !rec_triple_t
// CIR:   cir.copy %[[T_ARG]] align(4) to %{{.*}} align(4) : !cir.ptr<!rec_triple_t, target_address_space(5)>, !cir.ptr<!rec_triple_t>
// LLVM: define {{.*}}void @arg_budget(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr addrspace(5) noundef byref(%struct.triple_t) align 4 %[[T_ARG:.*]])
// LLVM:   call void @llvm.memcpy.{{.*}}(ptr{{.*}} align 4 %{{.*}}, ptr addrspace(5) align 4 %[[T_ARG]], i64 12, i1 false)

// Bit-fields are packed like any other small aggregate.
void arg_bits(bits_t b) {}

// CIR: cir.func {{.*}}@arg_bits(%[[B:[^:]+]]: !u32i{{.*}})
// CIR-NEXT:   %[[B_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!u32i, target_address_space(5)>
// CIR-NEXT:   cir.store %[[B]], %[[B_COERCE]] : !u32i, !cir.ptr<!u32i, target_address_space(5)>
// LLVM: define {{.*}}void @arg_bits(i32 %[[B:[^)]+]])
// LLVM:   store i32 %[[B]], ptr addrspace(5) %{{.*}}, align 4

// A _Complex up to 64 bits is packed like a small aggregate.
void arg_cfloat(_Complex float c) {}

// CIR: cir.func {{.*}}@arg_cfloat(%[[C:[^:]+]]: !cir.array<!u32i x 2> {{.*}})
// CIR-NEXT:   %[[C_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[C]], %[[C_COERCE]] : !cir.array<!u32i x 2>, !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)>
// CIR-NEXT:   cir.cast bitcast %[[C_COERCE]] : !cir.ptr<!cir.array<!u32i x 2>, target_address_space(5)> -> !cir.ptr<!cir.complex<!cir.float>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_cfloat([2 x i32] noundef %[[C:[^)]+]])
// LLVM:   store [2 x i32] %[[C]], ptr addrspace(5) %{{.*}}, align 4

// 128-bit integers pass whole.
void arg_int128(__int128 i) {}

// CIR: cir.func {{.*}}@arg_int128(%[[I:[^:]+]]: !s128i {{.*}})
// CIR:   cir.store %[[I]], %{{.*}} : !s128i, !cir.ptr<!s128i>
// LLVM: define {{.*}}void @arg_int128(i128 noundef %[[I:[^)]+]])
// LLVM:   store i128 %[[I]], ptr %{{.*}}

void arg_bitint128(_BitInt(128) i) {}

// CIR: cir.func {{.*}}@arg_bitint128(%[[I:[^:]+]]: !s128i_bitint {{.*}})
// CIR:   cir.store %[[I]], %{{.*}} : !s128i_bitint, !cir.ptr<!s128i_bitint>
// LLVM: define {{.*}}void @arg_bitint128(i128 noundef %[[I:[^)]+]])
// LLVM:   store i128 %[[I]], ptr %{{.*}}, align 8

// A call through a function pointer is coerced like a direct call.
void call_fnptr(void (*fp)(triple_t), triple_t t) { fp(t); }

// CIR: cir.func {{.*}}@call_fnptr(%{{.*}}: !cir.ptr<!cir.func<(!rec_triple_t)>> {{.*}}, %{{.*}}: !s32i{{.*}}, %{{.*}}: !s32i{{.*}}, %{{.*}}: !s32i{{.*}})
// CIR:   %[[FN:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!cir.func<(!rec_triple_t)>> -> !cir.ptr<!cir.func<(!s32i, !s32i, !s32i)>>
// CIR-NEXT:   cir.call %[[FN]](%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!cir.ptr<!cir.func<(!s32i, !s32i, !s32i)>>, !s32i, !s32i, !s32i) -> ()
// LLVM: define {{.*}}void @call_fnptr(ptr noundef %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
// LLVM:   %[[FN:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:   call void %[[FN]](i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

// A struct with a flexible array member is passed byval in addrspace(5).
void arg_fam(fam_t s) {}

// CIR: cir.func {{.*}}@arg_fam(%[[S_ARG:[^:]+]]: !cir.ptr<!rec_fam_t, target_address_space(5)> {llvm.align = 4 : i64, llvm.byval = !rec_fam_t, llvm.noundef}
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_t, target_address_space(5)>
// CIR-NEXT:   %[[S_VIEW:.*]] = cir.cast address_space %[[S]] : !cir.ptr<!rec_fam_t, target_address_space(5)> -> !cir.ptr<!rec_fam_t>
// CIR-NEXT:   cir.copy %[[S_ARG]] align(4) to %[[S_VIEW]] align(4) : !cir.ptr<!rec_fam_t, target_address_space(5)>, !cir.ptr<!rec_fam_t>
// LLVM: define {{.*}}void @arg_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})

// The caller passes its own local as the byval pointer.
void call_fam(fam_t s) { arg_fam(s); }

// CIR: cir.func {{.*}}@call_fam(
// CIR-NEXT:   %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_fam_t, target_address_space(5)>
// CIR:   cir.call @arg_fam(%[[S]]) {{.*}}: (!cir.ptr<!rec_fam_t, target_address_space(5)> {llvm.align = 4 : i64, llvm.byval = !rec_fam_t, llvm.noundef}) -> ()
// LLVM: define {{.*}}void @call_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})
// LLVM:   call void @arg_fam(ptr addrspace(5) noundef byval(%struct.fam_t) align 4 %{{.*}})

// A _Complex wider than 64 bits is split into its two parts.
void arg_cdouble(_Complex double c) {}

// CIR: cir.func {{.*}}@arg_cdouble(%[[RE:[^:]+]]: !cir.double {llvm.noundef}{{.*}}, %[[IM:[^:]+]]: !cir.double {llvm.noundef}{{.*}})
// CIR-NEXT:   %[[C_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<![[PAIR:rec_anon_struct[0-9]*]], target_address_space(5)>
// CIR-NEXT:   %[[C_RE:.*]] = cir.get_member %[[C_COERCE]][0] {name = ""} : !cir.ptr<![[PAIR]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   cir.store %[[RE]], %[[C_RE]] : !cir.double, !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[C_IM:.*]] = cir.get_member %[[C_COERCE]][1] {name = ""} : !cir.ptr<![[PAIR]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   cir.store %[[IM]], %[[C_IM]] : !cir.double, !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[C_PTR:.*]] = cir.cast bitcast %[[C_COERCE]] : !cir.ptr<![[PAIR]], target_address_space(5)> -> !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   %{{.*}} = cir.load %[[C_PTR]] : !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>, !cir.complex<!cir.double>
// LLVM: define {{.*}}void @arg_cdouble(double noundef %[[RE:[^,]+]], double noundef %[[IM:[^)]+]])
// LLVM:   %[[C_RE:.*]] = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %[[C:.*]], i32 0, i32 0
// LLVM-NEXT:   store double %[[RE]], ptr addrspace(5) %[[C_RE]], align 8
// LLVM-NEXT:   %[[C_IM:.*]] = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %[[C]], i32 0, i32 1
// LLVM-NEXT:   store double %[[IM]], ptr addrspace(5) %[[C_IM]], align 8

void arg_clong(_Complex long c) {}

// CIR: cir.func {{.*}}@arg_clong(%{{.*}}: !s64i {llvm.noundef}{{.*}}, %{{.*}}: !s64i {llvm.noundef}{{.*}})
// CIR:   cir.cast bitcast %{{.*}} : !cir.ptr<!rec_anon_struct{{[0-9]*}}, target_address_space(5)> -> !cir.ptr<!cir.complex<!s64i>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_clong(i64 noundef %[[RE:[^,]+]], i64 noundef %[[IM:[^)]+]])
// LLVM:   store i64 %[[RE]], ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   %{{.*}} = getelementptr inbounds nuw { i64, i64 }, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// LLVM-NEXT:   store i64 %[[IM]], ptr addrspace(5) %{{.*}}, align 8

// The caller splits it the same way.
void call_cdouble(_Complex double c) { arg_cdouble(c); }

// CIR: cir.func {{.*}}@call_cdouble(
// CIR:   %[[C_PAIR:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)> -> !cir.ptr<![[PAIR:rec_anon_struct[0-9]*]], target_address_space(5)>
// CIR-NEXT:   %[[C_RE:.*]] = cir.get_member %[[C_PAIR]][0] {name = ""} : !cir.ptr<![[PAIR]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[RE:.*]] = cir.load %[[C_RE]] : !cir.ptr<!cir.double, target_address_space(5)>, !cir.double
// CIR-NEXT:   %[[C_IM:.*]] = cir.get_member %[[C_PAIR]][1] {name = ""} : !cir.ptr<![[PAIR]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[IM:.*]] = cir.load %[[C_IM]] : !cir.ptr<!cir.double, target_address_space(5)>, !cir.double
// CIR-NEXT:   cir.call @arg_cdouble(%[[RE]], %[[IM]]) {{.*}}: (!cir.double {llvm.noundef}, !cir.double {llvm.noundef}) -> ()
// LLVM: define {{.*}}void @call_cdouble(double noundef %{{.*}}, double noundef %{{.*}})
// LLVM:   %[[RE:.*]] = load double, ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   %{{.*}} = getelementptr inbounds nuw { double, double }, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// LLVM-NEXT:   %[[IM:.*]] = load double, ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   call void @arg_cdouble(double noundef %[[RE]], double noundef %[[IM]])

// Past the register budget a _Complex is passed byref in addrspace(5).
void arg_cdouble_budget(int a0, int a1, int a2, int a3, int a4, int a5, int a6,
                        int a7, int a8, int a9, int a10, int a11, int a12,
                        int a13, _Complex double c) {}

// CIR: cir.func {{.*}}@arg_cdouble_budget(%arg0: !s32i {{.*}}%arg13: !s32i {{.*}}, %[[C_ARG:[^:]+]]: !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)> {llvm.align = 8 : i64, llvm.byref = !cir.complex<!cir.double>, llvm.noundef}
// CIR-NEXT:   %{{.*}} = cir.load %[[C_ARG]] : !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>, !cir.complex<!cir.double>
// LLVM: define {{.*}}void @arg_cdouble_budget(i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, i32 noundef %{{.*}}, ptr addrspace(5) noundef byref({ double, double }) align 8 %{{.*}})

// Through an ellipsis a _Complex is passed whole.
void call_var_cdouble(_Complex double c) { take_var(1, c); }

// CIR: cir.func {{.*}}@call_var_cdouble(%{{.*}}: !cir.double {llvm.noundef}{{.*}}, %{{.*}}: !cir.double {llvm.noundef}{{.*}})
// CIR:   cir.store align(8) %{{.*}}, %[[TMP:.*]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CIR-NEXT:   %[[C:.*]] = cir.load align(8) %[[TMP]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR-NEXT:   cir.call @take_var(%{{.*}}, %[[C]]) {{.*}}: (!s32i {llvm.noundef}, !cir.complex<!cir.double> {llvm.noundef}) -> ()
// LLVM: define {{.*}}void @call_var_cdouble(double noundef %{{.*}}, double noundef %{{.*}})
// LLVM:   call void (i32, ...) @take_var(i32 noundef 1, { double, double } noundef %{{.*}})

// A struct of one aggregate field is passed as that field.
void arg_cdwrap(cdwrap_t w) {}

// CIR: cir.func {{.*}}@arg_cdwrap(%[[W:[^:]+]]: !cir.complex<!cir.double>{{.*}})
// CIR-NEXT:   %[[W_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_cdwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_C:.*]] = cir.get_member %[[W_COERCE]][0] {name = ""} : !cir.ptr<!rec_cdwrap_t, target_address_space(5)> -> !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[W]], %[[W_C]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_cdwrap({ double, double } %{{.*}})

void arg_arrwrap(arrwrap_t w) {}

// CIR: cir.func {{.*}}@arg_arrwrap(%[[W:[^:]+]]: !cir.array<!s32i x 3>{{.*}})
// CIR-NEXT:   %[[W_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_arrwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_V:.*]] = cir.get_member %[[W_COERCE]][0] {name = ""} : !cir.ptr<!rec_arrwrap_t, target_address_space(5)> -> !cir.ptr<!cir.array<!s32i x 3>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[W]], %[[W_V]] : !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_arrwrap([3 x i32] %[[W:[^)]+]])
// LLVM:   %[[W_V:.*]] = getelementptr inbounds nuw %struct.arrwrap_t, ptr addrspace(5) %{{.*}}, i32 0, i32 0
// LLVM-NEXT:   store [3 x i32] %[[W]], ptr addrspace(5) %[[W_V]], align 4

void arg_nestwrap(nestwrap_t w) {}

// CIR: cir.func {{.*}}@arg_nestwrap(%[[W:[^:]+]]: ![[INNER:rec_anon[0-9A-Za-z]*]]
// CIR-NEXT:   %[[W_COERCE:.*]] = cir.alloca "coerce" align(4) : !cir.ptr<!rec_nestwrap_t, target_address_space(5)>
// CIR-NEXT:   %[[W_T:.*]] = cir.get_member %[[W_COERCE]][0] {name = ""} : !cir.ptr<!rec_nestwrap_t, target_address_space(5)> -> !cir.ptr<![[INNER]], target_address_space(5)>
// CIR-NEXT:   cir.store %[[W]], %[[W_T]] : ![[INNER]], !cir.ptr<![[INNER]], target_address_space(5)>
// LLVM: define {{.*}}void @arg_nestwrap(%struct.anon{{.*}} %{{.*}})

// A union is passed as its storage member, then its tail padding.
void arg_u_pad(u_pad_t u) {}

// CIR: cir.func {{.*}}@arg_u_pad(%[[D:[^:]+]]: !cir.double{{.*}}, %[[PAD:[^:]+]]: !cir.array<!u8i x 8>{{.*}})
// CIR-NEXT:   %[[U_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<![[FLAT:rec_anon_struct[0-9]*]], target_address_space(5)>
// CIR-NEXT:   %[[U_D:.*]] = cir.get_member %[[U_COERCE]][0] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   cir.store %[[D]], %[[U_D]] : !cir.double, !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[U_PAD:.*]] = cir.get_member %[[U_COERCE]][1] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!cir.array<!u8i x 8>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[PAD]], %[[U_PAD]] : !cir.array<!u8i x 8>, !cir.ptr<!cir.array<!u8i x 8>, target_address_space(5)>
// CIR-NEXT:   cir.cast bitcast %[[U_COERCE]] : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!rec_u_pad_t, target_address_space(5)>
// LLVM: define {{.*}}void @arg_u_pad(double %[[D:[^,]+]], [8 x i8] %[[PAD:[^)]+]])
// LLVM:   store double %[[D]], ptr addrspace(5) %{{.*}}, align 8
// LLVM-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// LLVM-NEXT:   store [8 x i8] %[[PAD]], ptr addrspace(5) %{{.*}}

void call_u_pad(u_pad_t u) { arg_u_pad(u); }

// CIR: cir.func {{.*}}@call_u_pad(
// CIR:   %[[U_FLAT:.*]] = cir.cast bitcast %{{.*}} : !cir.ptr<!rec_u_pad_t> -> !cir.ptr<![[FLAT:rec_anon_struct[0-9]*]]>
// CIR-NEXT:   %[[U_D:.*]] = cir.get_member %[[U_FLAT]][0] {name = ""} : !cir.ptr<![[FLAT]]> -> !cir.ptr<!cir.double>
// CIR-NEXT:   %[[D:.*]] = cir.load align(8) %[[U_D]] : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:   %[[U_PAD:.*]] = cir.get_member %[[U_FLAT]][1] {name = ""} : !cir.ptr<![[FLAT]]> -> !cir.ptr<!cir.array<!u8i x 8>>
// CIR-NEXT:   %[[PAD:.*]] = cir.load align(8) %[[U_PAD]] : !cir.ptr<!cir.array<!u8i x 8>>, !cir.array<!u8i x 8>
// CIR-NEXT:   cir.call @arg_u_pad(%[[D]], %[[PAD]]) {{.*}}: (!cir.double, !cir.array<!u8i x 8>) -> ()
// LLVM: define {{.*}}void @call_u_pad(double %{{.*}}, [8 x i8] %{{.*}})
// LLVM:   %[[D:.*]] = load double, ptr %{{.*}}, align 8
// LLVM-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr %{{.*}}, i32 0, i32 1
// LLVM-NEXT:   %[[PAD:.*]] = load [8 x i8], ptr %{{.*}}, align 8
// LLVM-NEXT:   call void @arg_u_pad(double %[[D]], [8 x i8] %[[PAD]])

// With no tail padding a union is passed as its storage member alone.
void arg_u_dl(u_dl_t u) {}

// CIR: cir.func {{.*}}@arg_u_dl(%[[U:[^:]+]]: !cir.array<!cir.double x 2>{{.*}})
// CIR-NEXT:   %[[U_COERCE:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<![[FLAT:rec_anon_struct[0-9]*]], target_address_space(5)>
// CIR-NEXT:   %[[U_D:.*]] = cir.get_member %[[U_COERCE]][0] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!cir.array<!cir.double x 2>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[U]], %[[U_D]] : !cir.array<!cir.double x 2>, !cir.ptr<!cir.array<!cir.double x 2>, target_address_space(5)>
// CIR-NEXT:   cir.cast bitcast %[[U_COERCE]] : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!rec_u_dl_t, target_address_space(5)>
// LLVM: define {{.*}}void @arg_u_dl([2 x double] %[[U:[^)]+]])
// LLVM:   store [2 x double] %[[U]], ptr addrspace(5) %{{.*}}, align 8

// A packed union keeps its padding, and its double is stored at align 1.
void arg_u_packed(u_packed_t u) {}

// CIR: cir.func {{.*}}@arg_u_packed(%[[D:[^:]+]]: !cir.double{{.*}}, %[[PAD:[^:]+]]: !cir.array<!u8i x 5>{{.*}})
// CIR-NEXT:   %[[U_COERCE:.*]] = cir.alloca "coerce" align(1) : !cir.ptr<![[FLAT:rec_anon_struct[0-9]*]], target_address_space(5)>
// CIR-NEXT:   %[[U_D:.*]] = cir.get_member %[[U_COERCE]][0] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   cir.store align(1) %[[D]], %[[U_D]] : !cir.double, !cir.ptr<!cir.double, target_address_space(5)>
// CIR-NEXT:   %[[U_PAD:.*]] = cir.get_member %[[U_COERCE]][1] {name = ""} : !cir.ptr<![[FLAT]], target_address_space(5)> -> !cir.ptr<!cir.array<!u8i x 5>, target_address_space(5)>
// CIR-NEXT:   cir.store %[[PAD]], %[[U_PAD]] : !cir.array<!u8i x 5>, !cir.ptr<!cir.array<!u8i x 5>, target_address_space(5)>
// LLVM: define {{.*}}void @arg_u_packed(double %[[D:[^,]+]], [5 x i8] %[[PAD:[^)]+]])
// LLVM:   store double %[[D]], ptr addrspace(5) %{{.*}}, align 1
// LLVM-NEXT:   %{{.*}} = getelementptr inbounds nuw {{.*}}, ptr addrspace(5) %{{.*}}, i32 0, i32 1
// LLVM-NEXT:   store [5 x i8] %[[PAD]], ptr addrspace(5) %{{.*}}, align 1
