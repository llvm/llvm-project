// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-SSE --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-SSE --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-SSE --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -fclangir -emit-cir %s -o %t-avx.cir
// RUN: FileCheck --check-prefixes=CIR,CIR-AVX --input-file=%t-avx.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -fclangir -emit-llvm %s -o %t-avx-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX --input-file=%t-avx-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm %s -o %t-avx.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-AVX --input-file=%t-avx.ll %s

typedef float v4f __attribute__((vector_size(16)));
typedef float v2f __attribute__((vector_size(8)));

typedef struct { unsigned short fam; char path[108]; } SockAddr;
typedef union { SockAddr s; void *err; } Expected;

typedef struct { void *a; void *b; unsigned c; } Large;
typedef union { char inlineRep[36]; Large large; } DenseMap;

typedef union { char c[17]; long l; } Odd24;
typedef struct { Odd24 u; _Bool checked : 1; } WrapsOdd24;
typedef struct { Odd24 a[2]; } ArrOfUnion;
typedef union { Odd24 inner; char pad[33]; } NestUnion;

typedef union { int x : 20; char buf[17]; } BitFieldBig;
typedef union { int i; } __attribute__((aligned(32))) OverAligned32;

typedef union { __float128 q; char c[17]; } Quad32;
typedef union { __float128 q; char c[49]; } Quad64;
typedef union { __float128 q; char c[65]; } Quad80;
typedef union { _Complex __float128 cq; char c[33]; } CplxQuad48;
typedef union { v4f v; char c[17]; } VecTailPad;
typedef union { v2f v; char c[17]; } NarrowVec;

// CIR-DAG: !rec_SockAddr = !cir.struct<"SockAddr" {data !u16i, data !cir.array<!s8i x 108>}>
// CIR-DAG: !rec_Large = !cir.struct<"Large" {data !cir.ptr<!void>, data !cir.ptr<!void>, data !u32i}>
// CIR-DAG: !rec_Expected = !cir.union<"Expected" {data !rec_SockAddr, data !cir.ptr<!void>}, padding = {!cir.array<!u8i x 104>}>
// CIR-DAG: !rec_DenseMap = !cir.union<"DenseMap" {data !cir.array<!s8i x 36>, data !rec_Large}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_Odd24 = !cir.union<"Odd24" {data !cir.array<!s8i x 17>, data !s64i}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_ArrOfUnion = !cir.struct<"ArrOfUnion" {data !cir.array<!rec_Odd24 x 2>}>
// CIR-DAG: !rec_NestUnion = !cir.union<"NestUnion" {data !rec_Odd24, data !cir.array<!s8i x 33>}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_BitFieldBig = !cir.union<"BitFieldBig" {bitfield !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!s32i, 20>]>, data !cir.array<!s8i x 17>}, padding = {!cir.array<!u8i x 3>}>
// CIR-DAG: !rec_OverAligned32 = !cir.union<"OverAligned32" {data !s32i}, padding = {!cir.array<!u8i x 28>}>
// CIR-DAG: !rec_Quad32 = !cir.union<"Quad32" {data !cir.f128, data !cir.array<!s8i x 17>}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_Quad64 = !cir.union<"Quad64" {data !cir.f128, data !cir.array<!s8i x 49>}, padding = {!cir.array<!u8i x 48>}>
// CIR-DAG: !rec_Quad80 = !cir.union<"Quad80" {data !cir.f128, data !cir.array<!s8i x 65>}, padding = {!cir.array<!u8i x 64>}>
// CIR-DAG: !rec_CplxQuad48 = !cir.union<"CplxQuad48" {data !cir.complex<!cir.f128>, data !cir.array<!s8i x 33>}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_VecTailPad = !cir.union<"VecTailPad" {data !cir.vector<4 x !cir.float>, data !cir.array<!s8i x 17>}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_NarrowVec = !cir.union<"NarrowVec" {data !cir.vector<2 x !cir.float>, data !cir.array<!s8i x 17>}, padding = {!cir.array<!u8i x 16>}>

// LLVM-DAG: %struct.Large = type { ptr, ptr, i32 }
// LLVM-DAG: %union.Expected = type { ptr, [104 x i8] }
// LLVM-DAG: %union.DenseMap = type { %struct.Large, [16 x i8] }
// LLVM-DAG: %union.Odd24 = type { i64, [16 x i8] }
// LLVM-DAG: %struct.WrapsOdd24 = type { %union.Odd24, i8 }
// LLVM-DAG: %struct.ArrOfUnion = type { [2 x %union.Odd24] }
// LLVM-DAG: %union.NestUnion = type { %union.Odd24, [16 x i8] }
// LLVM-DAG: %union.OverAligned32 = type { i32, [28 x i8] }
// LLVM-DAG: %union.Quad32 = type { fp128, [16 x i8] }
// LLVM-DAG: %union.Quad64 = type { fp128, [48 x i8] }
// LLVM-DAG: %union.Quad80 = type { fp128, [64 x i8] }
// LLVM-DAG: %union.CplxQuad48 = type { { fp128, fp128 }, [16 x i8] }
// LLVM-DAG: %union.VecTailPad = type { <4 x float>, [16 x i8] }
// LLVM-DAG: %union.NarrowVec = type { <2 x float>, [16 x i8] }

// 112 bytes against members of 110 and 8, so nothing spans the union.
void take_expected(Expected u) {}
// CIR: cir.func{{.*}} @take_expected(%arg0: !cir.ptr<!rec_Expected> {llvm.align = 8 : i64, llvm.byval = !rec_Expected, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_expected(ptr noundef byval(%union.Expected) align 8 %{{.+}})

// 40 bytes against members of 36 and 24.
void take_densemap(DenseMap u) {}
// CIR: cir.func{{.*}} @take_densemap(%arg0: !cir.ptr<!rec_DenseMap> {llvm.align = 8 : i64, llvm.byval = !rec_DenseMap, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_densemap(ptr noundef byval(%union.DenseMap) align 8 %{{.+}})

// 24 bytes against members of 17 and 8.
void take_odd24(Odd24 u) {}
// CIR: cir.func{{.*}} @take_odd24(%arg0: !cir.ptr<!rec_Odd24> {llvm.align = 8 : i64, llvm.byval = !rec_Odd24, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_odd24(ptr noundef byval(%union.Odd24) align 8 %{{.+}})

// The union reaches the classifier as a member of an enclosing struct.
void take_wraps_odd24(WrapsOdd24 u) {}
// CIR: cir.func{{.*}} @take_wraps_odd24(%arg0: !cir.ptr<!rec_WrapsOdd24> {llvm.align = 8 : i64, llvm.byval = !rec_WrapsOdd24, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_wraps_odd24(ptr noundef byval(%struct.WrapsOdd24) align 8 %{{.+}})

// Reached through an array member.
void take_arr_of_union(ArrOfUnion s) {}
// CIR: cir.func{{.*}} @take_arr_of_union(%arg0: !cir.ptr<!rec_ArrOfUnion> {llvm.align = 8 : i64, llvm.byval = !rec_ArrOfUnion, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_arr_of_union(ptr noundef byval(%struct.ArrOfUnion) align 8 %{{.+}})

// Reached through another union of the same kind.
void take_nest_union(NestUnion u) {}
// CIR: cir.func{{.*}} @take_nest_union(%arg0: !cir.ptr<!rec_NestUnion> {llvm.align = 8 : i64, llvm.byval = !rec_NestUnion, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_nest_union(ptr noundef byval(%union.NestUnion) align 8 %{{.+}})

// 20 bytes, with a named bit-field access unit no spanning member supplies
// data for.  Past two eightbytes the size settles that too.
void take_bitfield_big(BitFieldBig u) {}
// CIR: cir.func{{.*}} @take_bitfield_big(%arg0: !cir.ptr<!rec_BitFieldBig> {llvm.align = 8 : i64, llvm.byval = !rec_BitFieldBig, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_bitfield_big(ptr noundef byval(%union.BitFieldBig) align 8 %{{.+}})

// The declared alignment, not any member, is what put this past two
// eightbytes.
void take_over_aligned32(OverAligned32 u) {}
// CIR: cir.func{{.*}} @take_over_aligned32(%arg0: !cir.ptr<!rec_OverAligned32> {llvm.align = 32 : i64, llvm.byval = !rec_OverAligned32, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_over_aligned32(ptr noundef byval(%union.OverAligned32) align 32 %{{.+}})

// 32 bytes against members of 16 and 17.  The quad reaches SSEUP, so with AVX
// the coerce is named from the union's size, and without it memory.
void take_quad32(Quad32 u) {}
// CIR-SSE: cir.func{{.*}} @take_quad32(%arg0: !cir.ptr<!rec_Quad32> {llvm.align = 16 : i64, llvm.byval = !rec_Quad32, llvm.noundef} loc{{.*}})
// CIR-AVX: cir.func{{.*}} @take_quad32(%arg0: !cir.vector<4 x !cir.double> loc{{.*}})
// LLVM-SSE: define{{.*}} void @take_quad32(ptr noundef byval(%union.Quad32) align 16 %{{.+}})
// LLVM-AVX: define{{.*}} void @take_quad32(<4 x double> %{{.+}})

// 64 bytes, the widest size an SSEUP coerce can be named from.
void take_quad64(Quad64 u) {}
// CIR-SSE: cir.func{{.*}} @take_quad64(%arg0: !cir.ptr<!rec_Quad64> {llvm.align = 16 : i64, llvm.byval = !rec_Quad64, llvm.noundef} loc{{.*}})
// CIR-AVX: cir.func{{.*}} @take_quad64(%arg0: !cir.vector<8 x !cir.double> loc{{.*}})
// LLVM-SSE: define{{.*}} void @take_quad64(ptr noundef byval(%union.Quad64) align 16 %{{.+}})
// LLVM-AVX: define{{.*}} void @take_quad64(<8 x double> %{{.+}})

// 80 bytes, past 512, so the quad member cannot put it in registers.
void take_quad80(Quad80 u) {}
// CIR: cir.func{{.*}} @take_quad80(%arg0: !cir.ptr<!rec_Quad80> {llvm.align = 16 : i64, llvm.byval = !rec_Quad80, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_quad80(ptr noundef byval(%union.Quad80) align 16 %{{.+}})

// A complex quad reaches SSEUP nowhere, so its 48 bytes are not refused.
void take_cplx_quad48(CplxQuad48 u) {}
// CIR: cir.func{{.*}} @take_cplx_quad48(%arg0: !cir.ptr<!rec_CplxQuad48> {llvm.align = 16 : i64, llvm.byval = !rec_CplxQuad48, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_cplx_quad48(ptr noundef byval(%union.CplxQuad48) align 16 %{{.+}})

// A vector reaches SSEUP too, here with nothing spanning the union.
void take_vec_tail_pad(VecTailPad u) {}
// CIR-SSE: cir.func{{.*}} @take_vec_tail_pad(%arg0: !cir.ptr<!rec_VecTailPad> {llvm.align = 16 : i64, llvm.byval = !rec_VecTailPad, llvm.noundef} loc{{.*}})
// CIR-AVX: cir.func{{.*}} @take_vec_tail_pad(%arg0: !cir.vector<4 x !cir.double> loc{{.*}})
// LLVM-SSE: define{{.*}} void @take_vec_tail_pad(ptr noundef byval(%union.VecTailPad) align 16 %{{.+}})
// LLVM-AVX: define{{.*}} void @take_vec_tail_pad(<4 x double> %{{.+}})

// A vector narrower than 128 bits never reaches SSEUP, so this union is
// classified from its size at every target.
void take_narrow_vec(NarrowVec u) {}
// CIR: cir.func{{.*}} @take_narrow_vec(%arg0: !cir.ptr<!rec_NarrowVec> {llvm.align = 8 : i64, llvm.byval = !rec_NarrowVec, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @take_narrow_vec(ptr noundef byval(%union.NarrowVec) align 8 %{{.+}})

Expected ret_expected(Expected u) { return u; }
// CIR: cir.func{{.*}} @ret_expected(%arg0: !cir.ptr<!rec_Expected> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Expected, llvm.writable} loc{{.*}}, %arg1: !cir.ptr<!rec_Expected> {llvm.align = 8 : i64, llvm.byval = !rec_Expected, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @ret_expected(ptr dead_on_unwind noalias writable sret(%union.Expected) align 8 %{{[^,]+}}, ptr noundef byval(%union.Expected) align 8 %{{.+}})

Odd24 ret_odd24(Odd24 u) { return u; }
// CIR: cir.func{{.*}} @ret_odd24(%arg0: !cir.ptr<!rec_Odd24> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Odd24, llvm.writable} loc{{.*}}, %arg1: !cir.ptr<!rec_Odd24> {llvm.align = 8 : i64, llvm.byval = !rec_Odd24, llvm.noundef} loc{{.*}})
// LLVM: define{{.*}} void @ret_odd24(ptr dead_on_unwind noalias writable sret(%union.Odd24) align 8 %{{[^,]+}}, ptr noundef byval(%union.Odd24) align 8 %{{.+}})

void call_odd24(Odd24 u) { take_odd24(u); }
// CIR: cir.func{{.*}} @call_odd24(%arg0: !cir.ptr<!rec_Odd24> {llvm.align = 8 : i64, llvm.byval = !rec_Odd24, llvm.noundef} loc{{.*}})
// CIR:   %[[SLOT:.*]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_Odd24>
// CIR:   cir.call @take_odd24(%[[SLOT]])
// LLVM: define{{.*}} void @call_odd24(ptr noundef byval(%union.Odd24) align 8 %{{.+}})
// LLVM:   call void @take_odd24(ptr noundef byval(%union.Odd24) align 8 %{{.+}})
