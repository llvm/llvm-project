// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// A vector of _BitInt(N) lowers to a vector of iN in both its value and its
// memory form.

typedef _BitInt(32) v4s32 __attribute__((ext_vector_type(4)));
typedef unsigned _BitInt(16) v4u16 __attribute__((ext_vector_type(4)));
typedef unsigned _BitInt(8) v8u8 __attribute__((ext_vector_type(8)));
typedef _BitInt(128) v2s128 __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));

v4s32 g32 = {1, 2, -3, 4};
v8u8 g8 = {0, 1, 2, 3, 252, 253, 254, 255};
v2s128 g128;

// CIR-DAG: cir.global external @g32 = #cir.const_vector<[#cir.int<1> : !s32i_bitint, #cir.int<2> : !s32i_bitint, #cir.int<-3> : !s32i_bitint, #cir.int<4> : !s32i_bitint]> : !cir.vector<4 x !s32i_bitint> align(16)
// CIR-DAG: cir.global external @g8 = #cir.const_vector<[#cir.int<0> : !u8i_bitint, #cir.int<1> : !u8i_bitint, #cir.int<2> : !u8i_bitint, #cir.int<3> : !u8i_bitint, #cir.int<252> : !u8i_bitint, #cir.int<253> : !u8i_bitint, #cir.int<254> : !u8i_bitint, #cir.int<255> : !u8i_bitint]> : !cir.vector<8 x !u8i_bitint> align(8)
// CIR-DAG: cir.global external @g128 = #cir.zero : !cir.vector<2 x !s128i_bitint> align(16)

// LLVM-DAG: @g32 = global <4 x i32> <i32 1, i32 2, i32 -3, i32 4>, align 16
// LLVM-DAG: @g8 = global <8 x i8> <i8 0, i8 1, i8 2, i8 3, i8 -4, i8 -3, i8 -2, i8 -1>, align 8
// LLVM-DAG: @g128 = global <2 x i128> zeroinitializer, align 16

void arith(v8u8 *p, v2s128 *q) {
  *p = *p * *p;
  v2s128 v = *q;
  v[1] = v[0] + 3;
  *q = v;
}

// CIR-LABEL: cir.func {{.*}} @arith
// CIR:   cir.load {{.*}} : !cir.ptr<!cir.vector<8 x !u8i_bitint>>, !cir.vector<8 x !u8i_bitint>
// CIR:   cir.mul %{{.*}}, %{{.*}} : !cir.vector<8 x !u8i_bitint>
// CIR:   cir.vec.extract %{{.*}}[%{{.*}} : !s32i] : !cir.vector<2 x !s128i_bitint>
// CIR:   cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !s32i] : !cir.vector<2 x !s128i_bitint>

// LLVM-LABEL: define {{.*}} void @arith
// LLVM:   %[[A:.*]] = load <8 x i8>, ptr %{{.*}}, align 8
// LLVM:   %[[B:.*]] = load <8 x i8>, ptr %{{.*}}, align 8
// LLVM:   %[[MUL:.*]] = mul <8 x i8> %[[A]], %[[B]]
// LLVM:   store <8 x i8> %[[MUL]], ptr %{{.*}}, align 8
// LLVM:   %[[V:.*]] = load <2 x i128>, ptr %{{.*}}, align 16
// LLVM:   store <2 x i128> %[[V]], ptr %{{.*}}, align 16
// LLVM:   %[[V0:.*]] = extractelement <2 x i128> %{{.*}}, i32 0
// LLVM:   %[[ADD:.*]] = add nsw i128 %[[V0]], 3
// LLVM:   insertelement <2 x i128> %{{.*}}, i128 %[[ADD]], i32 1

v4s32 select_min(v4s32 a, v4s32 b) { return a < b ? a : b; }

// CIR-LABEL: cir.func {{.*}} @select_min(%{{.*}}: !cir.vector<4 x !s32i_bitint> {{.*}}, %{{.*}}: !cir.vector<4 x !s32i_bitint> {{.*}}) -> !cir.vector<4 x !s32i_bitint>
// CIR:   cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<4 x !s32i_bitint>, !cir.vector<4 x !s32i>
// CIR:   cir.vec.ternary(%{{.*}}, %{{.*}}, %{{.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i_bitint>

// LLVM-LABEL: define {{.*}} <4 x i32> @select_min(<4 x i32> {{.*}}, <4 x i32> {{.*}})
// LLVM:   %[[CMP:.*]] = icmp slt <4 x i32> %{{.*}}, %{{.*}}
// LLVM:   %[[MASK:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
// LLVMCIR: %[[COND:.*]] = icmp ne <4 x i32> %[[MASK]], zeroinitializer
// OGCG:    %[[COND:.*]] = icmp slt <4 x i32> %[[MASK]], zeroinitializer
// LLVM:   select <4 x i1> %[[COND]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}}

v4f convert(v4s32 a) {
  v4u16 t = __builtin_convertvector(a, v4u16);
  return __builtin_convertvector(t, v4f);
}

// CIR-LABEL: cir.func {{.*}} @convert
// CIR:   cir.cast integral %{{.*}} : !cir.vector<4 x !s32i_bitint> -> !cir.vector<4 x !u16i_bitint>
// CIR:   cir.cast int_to_float %{{.*}} : !cir.vector<4 x !u16i_bitint> -> !cir.vector<4 x !cir.float>

// LLVM-LABEL: define {{.*}} <4 x float> @convert(<4 x i32> {{.*}})
// LLVM:   %[[T:.*]] = trunc <4 x i32> %{{.*}} to <4 x i16>
// LLVM:   store <4 x i16> %[[T]], ptr %{{.*}}, align 8
// LLVM:   %[[T2:.*]] = load <4 x i16>, ptr %{{.*}}, align 8
// LLVM:   uitofp <4 x i16> %[[T2]] to <4 x float>
