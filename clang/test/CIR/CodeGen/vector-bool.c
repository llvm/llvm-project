// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports vector types.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// A packed boolean vector is held in registers as one lane per element, but
// stored as an integer with one bit per lane, padded up to at least a byte.

typedef _Bool vb4 __attribute__((ext_vector_type(4)));
typedef _Bool vb8 __attribute__((ext_vector_type(8)));
typedef _Bool vb16 __attribute__((ext_vector_type(16)));

vb4 g4;
vb8 g8;
vb16 g16;

// CIR: cir.global external @g4 = #cir.int<0> : !u8i {alignment = 1 : i64}
// CIR: cir.global external @g8 = #cir.int<0> : !u8i {alignment = 1 : i64}
// CIR: cir.global external @g16 = #cir.int<0> : !u16i {alignment = 2 : i64}

// LLVM: @g4 = global i8 0, align 1
// LLVM: @g8 = global i8 0, align 1
// LLVM: @g16 = global i16 0, align 2

// OGCG: @g4 = global i8 0, align 1
// OGCG: @g8 = global i8 0, align 1
// OGCG: @g16 = global i16 0, align 2

void store(vb8 *p, vb8 v) { *p = v; }

// CIR-LABEL: cir.func {{.*}}@store
// CIR:         %[[SLOT:.*]] = cir.alloca "v" align(1) init : !cir.ptr<!u8i>
// CIR:         %[[PACKED:.*]] = cir.cast bitcast %{{.*}} : !cir.vector<8 x !cir.bool> -> !u8i
// CIR:         cir.store %[[PACKED]], %[[SLOT]] : !u8i, !cir.ptr<!u8i>
// CIR:         %[[BITS:.*]] = cir.load align(1) %[[SLOT]] : !cir.ptr<!u8i>, !u8i
// CIR:         %[[VEC:.*]] = cir.cast bitcast %[[BITS]] : !u8i -> !cir.vector<8 x !cir.bool>
// CIR:         %[[OUT:.*]] = cir.cast bitcast %[[VEC]] : !cir.vector<8 x !cir.bool> -> !u8i
// CIR:         cir.store align(1) %[[OUT]], %{{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-LABEL: define {{.*}}void @store
// LLVM:         %[[SLOT:.*]] = alloca i8, i64 1, align 1
// LLVM:         %[[PACKED:.*]] = bitcast <8 x i1> %{{.*}} to i8
// LLVM:         store i8 %[[PACKED]], ptr %[[SLOT]], align 1
// LLVM:         %[[BITS:.*]] = load i8, ptr %[[SLOT]], align 1
// LLVM:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// LLVM:         %[[OUT:.*]] = bitcast <8 x i1> %[[VEC]] to i8
// LLVM:         store i8 %[[OUT]], ptr %{{.*}}, align 1

// OGCG-LABEL: define {{.*}}void @store
// OGCG:         %[[PACKED:.*]] = bitcast <8 x i1> %{{.*}} to i8
// OGCG:         store i8 %[[PACKED]], ptr %[[SLOT:.*]], align 1
// OGCG:         %[[BITS:.*]] = load i8, ptr %[[SLOT]], align 1
// OGCG:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// OGCG:         %[[OUT:.*]] = bitcast <8 x i1> %[[VEC]] to i8
// OGCG:         store i8 %[[OUT]], ptr %{{.*}}, align 1

vb8 load(vb8 *p) { return *p; }

// The return slot holds the value representation, so it is a vector of lanes.

// CIR-LABEL: cir.func {{.*}}@load
// CIR:         %[[RET:.*]] = cir.alloca "__retval" align(1) : !cir.ptr<!cir.vector<8 x !cir.bool>>
// CIR:         %[[BITS:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!u8i>, !u8i
// CIR:         %[[VEC:.*]] = cir.cast bitcast %[[BITS]] : !u8i -> !cir.vector<8 x !cir.bool>
// CIR:         cir.store %[[VEC]], %[[RET]] : !cir.vector<8 x !cir.bool>, !cir.ptr<!cir.vector<8 x !cir.bool>>

// LLVM-LABEL: define {{.*}}<8 x i1> @load
// LLVM:         %[[RET:.*]] = alloca <8 x i1>, i64 1, align 1
// LLVM:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// LLVM:         store <8 x i1> %[[VEC]], ptr %[[RET]], align 1

// OGCG-LABEL: define {{.*}}i8 @load
// OGCG:         %[[RET:.*]] = alloca <8 x i1>, align 1
// OGCG:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// OGCG:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// OGCG:         store <8 x i1> %[[VEC]], ptr %[[RET]], align 1

// A vector with fewer than 8 lanes is padded out to a byte, so moving it in and
// out of memory reshapes it with a shuffle.

void store4(vb4 *p, vb4 v) { *p = v; }

// CIR-LABEL: cir.func {{.*}}@store4
// CIR:         %[[WIDE:.*]] = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.bool>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i, #cir.int<-1> : !s32i] : !cir.vector<8 x !cir.bool>
// CIR:         %[[PACKED:.*]] = cir.cast bitcast %[[WIDE]] : !cir.vector<8 x !cir.bool> -> !u8i
// CIR:         cir.store %[[PACKED]], %{{.*}} : !u8i, !cir.ptr<!u8i>
// CIR:         %[[BITS:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!u8i>, !u8i
// CIR:         %[[PADDED:.*]] = cir.cast bitcast %[[BITS]] : !u8i -> !cir.vector<8 x !cir.bool>
// CIR:         %[[NARROW:.*]] = cir.vec.shuffle(%[[PADDED]], %{{.*}} : !cir.vector<8 x !cir.bool>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.bool>

// LLVM-LABEL: define {{.*}}void @store4
// LLVM:         %[[WIDE:.*]] = shufflevector <4 x i1> %{{.*}}, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
// LLVM:         %[[PACKED:.*]] = bitcast <8 x i1> %[[WIDE]] to i8
// LLVM:         store i8 %[[PACKED]], ptr %{{.*}}, align 1
// LLVM:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM:         %[[PADDED:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// LLVM:         %[[NARROW:.*]] = shufflevector <8 x i1> %[[PADDED]], <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

// OGCG-LABEL: define {{.*}}void @store4
// OGCG:         %[[WIDE:.*]] = shufflevector <4 x i1> %{{.*}}, <4 x i1> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison>
// OGCG:         %[[PACKED:.*]] = bitcast <8 x i1> %[[WIDE]] to i8
// OGCG:         store i8 %[[PACKED]], ptr %{{.*}}, align 1
// OGCG:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// OGCG:         %[[PADDED:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// OGCG:         %[[NARROW:.*]] = shufflevector <8 x i1> %[[PADDED]], <8 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

_Bool extract(vb8 *p) { return (*p)[3]; }

// CIR-LABEL: cir.func {{.*}}@extract
// CIR:         %[[BITS:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!u8i>, !u8i
// CIR:         %[[VEC:.*]] = cir.cast bitcast %[[BITS]] : !u8i -> !cir.vector<8 x !cir.bool>
// CIR:         %{{.*}} = cir.vec.extract %[[VEC]][%{{.*}} : !s32i] : !cir.vector<8 x !cir.bool>

// LLVM-LABEL: define {{.*}}i1 @extract
// LLVM:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// LLVM:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// LLVM:         %{{.*}} = extractelement <8 x i1> %[[VEC]], i32 3

// OGCG-LABEL: define {{.*}}i1 @extract
// OGCG:         %[[BITS:.*]] = load i8, ptr %{{.*}}, align 1
// OGCG:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// OGCG:         %{{.*}} = extractelement <8 x i1> %[[VEC]], i32 3

// Writing one lane is a read/modify/write of the whole packed integer.

void insert(vb8 *p, _Bool b) { (*p)[2] = b; }

// CIR-LABEL: cir.func {{.*}}@insert
// CIR:         %[[BITS:.*]] = cir.load align(1) %{{.*}} : !cir.ptr<!u8i>, !u8i
// CIR:         %[[VEC:.*]] = cir.cast bitcast %[[BITS]] : !u8i -> !cir.vector<8 x !cir.bool>
// CIR:         %[[NEW:.*]] = cir.vec.insert %{{.*}}, %[[VEC]][%{{.*}} : !s32i] : !cir.vector<8 x !cir.bool>
// CIR:         %[[OUT:.*]] = cir.cast bitcast %[[NEW]] : !cir.vector<8 x !cir.bool> -> !u8i
// CIR:         cir.store align(1) %[[OUT]], %{{.*}} : !u8i, !cir.ptr<!u8i>

// LLVM-LABEL: define {{.*}}void @insert
// LLVM:         %[[PTR:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         %[[BITS:.*]] = load i8, ptr %[[PTR]], align 1
// LLVM:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// LLVM:         %[[NEW:.*]] = insertelement <8 x i1> %[[VEC]], i1 %{{.*}}, i32 2
// LLVM:         %[[OUT:.*]] = bitcast <8 x i1> %[[NEW]] to i8
// LLVM:         store i8 %[[OUT]], ptr %{{.*}}, align 1

// OGCG-LABEL: define {{.*}}void @insert
// OGCG:         %[[PTR:.*]] = load ptr, ptr %{{.*}}, align 8
// OGCG:         %[[BITS:.*]] = load i8, ptr %[[PTR]], align 1
// OGCG:         %[[VEC:.*]] = bitcast i8 %[[BITS]] to <8 x i1>
// OGCG:         %[[NEW:.*]] = insertelement <8 x i1> %[[VEC]], i1 %{{.*}}, i32 2
// OGCG:         %[[OUT:.*]] = bitcast <8 x i1> %[[NEW]] to i8
// OGCG:         store i8 %[[OUT]], ptr %{{.*}}, align 1
