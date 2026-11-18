// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=A64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=A64-LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=A64-LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=X64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t.ll %s

// The vector alignment cap is TargetInfo::getMaxVectorAlign(). AArch64 sets
// it to 128 bits, so a vector wider than 16 bytes is 16-byte aligned.
// x86_64 leaves the cap at 0, so the same vector keeps its natural alignment
// and the module records no #cir.vector_spec entry.

typedef float v4f32 __attribute__((vector_size(16)));
typedef float v8f32 __attribute__((vector_size(32)));
typedef char v17i8 __attribute__((vector_size(17)));

v4f32 g4 = {0};
v8f32 g8 = {0};
v17i8 g17 = {0};

// A64-CIR-DAG: !cir.vector<1 x !cir.int<u, 8>> = #cir.vector_spec<max_align = 128>
// A64-CIR-DAG: cir.global external @g4 = {{.*}}!cir.vector<4 x !cir.float> align(16)
// A64-CIR-DAG: cir.global external @g8 = {{.*}}!cir.vector<8 x !cir.float> align(16)
// A64-CIR-DAG: cir.global external @g17 = {{.*}}!cir.vector<17 x !s8i> align(16)
// A64-LLVM-DAG: @g4 = {{.*}}global <4 x float> zeroinitializer, align 16
// A64-LLVM-DAG: @g8 = {{.*}}global <8 x float> zeroinitializer, align 16
// A64-LLVM-DAG: @g17 = {{.*}}global <17 x i8> zeroinitializer, align 16

// X64-CIR-NOT: #cir.vector_spec
// X64-CIR-DAG: cir.global external @g4 = {{.*}}!cir.vector<4 x !cir.float> align(16)
// X64-CIR-DAG: cir.global external @g8 = {{.*}}!cir.vector<8 x !cir.float> align(32)
// X64-CIR-DAG: cir.global external @g17 = {{.*}}!cir.vector<17 x !s8i> align(32)
// X64-LLVM-DAG: @g4 = {{.*}}global <4 x float> zeroinitializer, align 16
// X64-LLVM-DAG: @g8 = {{.*}}global <8 x float> zeroinitializer, align 32
// X64-LLVM-DAG: @g17 = {{.*}}global <17 x i8> zeroinitializer, align 32

// On AArch64 the cap lowers the C alignment of a vector wider than 16 bytes
// but not the alignment of the LLVM vector type. A record holding such a
// vector is laid out with the C alignment, so CIR and classic codegen both
// lower it to a packed LLVM struct with explicit padding. A load of a member
// assumes the record's 16-byte alignment.

typedef float v9f32 __attribute__((ext_vector_type(9)));

struct S { int i; v8f32 v; };
struct T { v9f32 a; int b; };
struct In { v8f32 v; };
struct W { int i; struct In s; };
struct A { v9f32 a[2]; int x; };
struct V { v9f32 a; };

v9f32 ga[2];
v9f32 *gp = &ga[1];

// A64-CIR-DAG: !rec_S = !cir.struct<"S" packed {data !s32i, pad !cir.array<!u8i x 12>, data !cir.vector<8 x !cir.float>}>
// A64-CIR-DAG: !rec_T = !cir.struct<"T" packed {data !cir.vector<9 x !cir.float>, data !s32i, pad !cir.array<!u8i x 12>}>
// A64-CIR-DAG: !rec_In = !cir.struct<"In" {data !cir.vector<8 x !cir.float>}>
// A64-CIR-DAG: !rec_W = !cir.struct<"W" packed {data !s32i, pad !cir.array<!u8i x 12>, data !rec_In}>
// A64-CIR-DAG: !rec_A = !cir.struct<"A" packed {data !cir.array<!cir.vector<9 x !cir.float> x 2>, data !s32i, pad !cir.array<!u8i x 12>}>
// A64-CIR-DAG: !rec_V = !cir.struct<"V" {data !cir.vector<9 x !cir.float>}>
// A64-CIR-DAG: cir.global external @ga = #cir.zero : !cir.array<!cir.vector<9 x !cir.float> x 2> align(16)
// A64-CIR-DAG: cir.global external @gp = #cir.global_view<@ga, [1 : i32]> : !cir.ptr<!cir.vector<9 x !cir.float>> align(8)

// A64-LLVM-DAG: %struct.S = type <{ i32, [12 x i8], <8 x float> }>
// A64-LLVM-DAG: %struct.T = type <{ <9 x float>, i32, [12 x i8] }>
// A64-LLVM-DAG: %struct.In = type { <8 x float> }
// A64-LLVM-DAG: %struct.W = type <{ i32, [12 x i8], %struct.In }>
// A64-LLVM-DAG: %struct.A = type <{ [2 x <9 x float>], i32, [12 x i8] }>
// A64-LLVM-DAG: @ga = {{.*}}global [2 x <9 x float>] zeroinitializer, align 16
// A64-LLVM-DAG: @gp = {{.*}}global ptr getelementptr {{.*}}(i8, ptr @ga, i64 64), align 8

float readS(struct S *p) { return p->v[0]; }

// A64-LLVM-LABEL: define {{.*}}float @readS(
// A64-LLVM:         %[[V:.*]] = getelementptr inbounds nuw %struct.S, ptr %{{.*}}, i32 0, i32 2
// A64-LLVM-NEXT:    load <8 x float>, ptr %[[V]], align 16

int readT(struct T *p) { return p->b; }

// A64-LLVM-LABEL: define {{.*}}i32 @readT(
// A64-LLVM:         %[[B:.*]] = getelementptr inbounds nuw %struct.T, ptr %{{.*}}, i32 0, i32 1
// A64-LLVM-NEXT:    load i32, ptr %[[B]], align 16

float readW(struct W *p) { return p->s.v[0]; }

// A64-LLVM-LABEL: define {{.*}}float @readW(
// A64-LLVM:         %[[S:.*]] = getelementptr inbounds nuw %struct.W, ptr %{{.*}}, i32 0, i32 2
// A64-LLVM-NEXT:    %[[V:.*]] = getelementptr inbounds nuw %struct.In, ptr %[[S]], i32 0, i32 0
// A64-LLVM-NEXT:    load <8 x float>, ptr %[[V]], align 16

int readA(struct A *p) { return p->x; }

// A64-LLVM-LABEL: define {{.*}}i32 @readA(
// A64-LLVM:         %[[X:.*]] = getelementptr inbounds nuw %struct.A, ptr %{{.*}}, i32 0, i32 1
// A64-LLVM-NEXT:    load i32, ptr %[[X]], align 16

// The allocation size of a v9f32 is 64 bytes, its 36-byte size rounded up to
// the alignment of <9 x float>, not to the capped C alignment.

long diff(v9f32 *a, v9f32 *b) { return a - b; }

// A64-LLVM-LABEL: define {{.*}}i64 @diff(
// A64-LLVM:         sdiv exact i64 %{{.*}}, 64

void copyV(struct V *d, struct V *s) { *d = *s; }

// A64-LLVM-LABEL: define {{.*}}void @copyV(
// A64-LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{.*}}, ptr align 16 %{{.*}}, i64 64, i1 false)
