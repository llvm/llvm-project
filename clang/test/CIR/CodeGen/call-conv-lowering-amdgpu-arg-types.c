// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Checks that CallConvLowering classifies arguments of ordinary AMDGPU
// functions, which use the default convention rather than the kernel one.

// TODO(cir): Add the argument cases still NYI in CallConvLowering:
// - Aggregates of 33 to 64 bits, which classic coerces to [2 x i32].
//   abiTypeToCIR has no array case yet.
// - Aggregates over 64 bits that fit the 16-register budget. The classifier
//   returns Direct with no coerce type, which the bridge rejects.
// - Aggregates past the register budget, passed byref in addrspace(5).
// - Structs with a flexible array member and transparent unions. The bridge
//   does not set those record flags yet.
// - Arguments passed through an ellipsis.

typedef struct {} empty_t;
typedef struct { char c; } i8_wrapper_t;
typedef struct { int i; } i32_wrapper_t;
typedef struct { float f; } f32_wrapper_t;
typedef struct { int *p; } ptr_wrapper_t;
typedef struct { char a, b; } char2_t;
typedef struct { char a, b, c; } small_t;
typedef union { int i; float f; } int_float_t;
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

// The call site is coerced the same way as the callee.
void call_small(small_t s) { arg_small(s); }

// CIR: cir.func {{.*}}@call_small(%arg0: !u32i{{.*}})
// CIR:   cir.call @arg_small(%{{.*}}){{.*}}: (!u32i{{.*}}) -> ()
// LLVM: define {{.*}}void @call_small(i32 %{{.*}})
// LLVM:   call void @arg_small(i32 %{{.*}})

// The call site carries the same extension as the callee.
void call_char(char c) { arg_char(c); }

// LLVM: define {{.*}}void @call_char(i8 noundef signext %{{.*}})
// LLVM:   call void @arg_char(i8 noundef signext %{{.*}})
